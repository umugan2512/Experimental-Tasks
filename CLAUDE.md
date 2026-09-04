# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is **`Experimental-Tasks`**, the combined repo this lab actually works out of day to day:
PyBpod itself (a GUI application, built on PyForms/Qt, for controlling Bpod behavioral-experiment
hardware from Sanworks, maintained by the Champalimaud Foundation) as a git submodule, plus this
lab's own `_projects/` — real task code, training scripts, and the shared `training_log.xlsx` —
committed alongside it so the whole stack transfers cleanly across every behavior-box computer with
one `git clone` + `git submodule update --init --recursive`.

- `pybpod/` — the PyBpod application itself, as a git submodule pointed at **this lab's own fork**
  (`https://github.com/umugan2512/pybpod.git`, branch `save-local-changes`), not the official
  upstream (`github.com/pybpod/pybpod`, no push access there). See "Forked submodules" below for
  why, and for the same pattern applied one level deeper for the rotary encoder plugin.
- `_projects/` — this lab's real task code and training data (see "Local project data" below).
  Moved in as a real, mostly-committed part of *this* repo — no longer a separate, ungit'd sibling
  directory the way it was in the original single-machine `pybpod` checkout this was migrated from.
- `Calibration/` — hardware calibration tooling shared across every project (currently: liquid
  reward valve calibration; see "Liquid reward (valve) calibration" below) — a sibling of
  `_projects/`, not nested inside any single `_projects/<ProjectName>/`, since a rig's own physical
  calibration isn't specific to one experiment. The first top-level folder of this kind in the repo.
- `CLAUDE.md` (this file) — lives at this repo's own top level, not inside the `pybpod/` submodule,
  even though most of its historical content (the sections below down through "Architecture notes")
  was originally written for `pybpod` on its own. It now documents the whole combined repo.

### Submodules are not checked out by default

`pybpod/` itself, and everything under `pybpod/base/`, `pybpod/libraries/`, `pybpod/plugins/`, are
git submodules (nested — `pybpod` is `Experimental-Tasks`'s own submodule, and it in turn has ~17 of
its own). If any of these appear as **empty directories**, they have not been initialized — the
normal state right after a plain `git clone`. Before expecting to find real source code under those
paths, run:

```bash
git submodule update --init --recursive
```

Do not assume a submodule is "missing" or "broken" just because its folder is empty — check
submodule init status first (`git submodule status`, entries prefixed with `-` are uninitialized).
Confirmed the hard way: forgetting this step after the initial `Experimental-Tasks` migration left
every plugin folder (including the rotary encoder module, whose own local fixes this lab depends on
for hardware to work at all) completely empty, causing a real `AttributeError:
'RotaryEncoderModule' has no attribute 'discover'` crash mid-session.

## Forked submodules (pybpod itself, and the rotary encoder plugin)

Two submodules in this tree carry **real local changes this lab depends on** that don't exist
upstream, so both are pinned to a fork (not the official repo) on a branch literally named
`save-local-changes`:

- `pybpod` itself → `github.com/umugan2512/pybpod.git` — carries this repo's own `CLAUDE.md` (now
  superseded by the top-level copy, kept for history), `utils/install.py`'s local edit, and the
  from-scratch `pybpod-gui-plugin-hifi` package.
- `pybpod/plugins/pybpod-gui-plugin-rotaryencoder` → its own separate fork,
  `github.com/umugan2512/pybpod-gui-plugin-rotaryencoder.git` — carries `RotaryEncoderModule.discover()`
  (handshake-probes serial ports since a module's own USB connection is never reported over the
  Bpod relay — see "Hardware module notes" below), `BOOT_DELAY_S`/`HANDSHAKE_RETRIES`, retry-verified
  `set_zero_position()`/`set_thresholds()`, and `wheel_position_plot.py` (the "Wheel Position"
  live-plot GUI panel). **This whole plugin's local changes were originally uncommitted, vendored
  edits sitting directly in the submodule's working tree** — easy to lose entirely on any repo
  restructuring that doesn't specifically account for it (confirmed: got missed once already during
  the `Experimental-Tasks` migration, since a plain `git status` on the *parent* repo only shows a
  submodule as generically "dirty," not what's actually different inside it). **Before assuming any
  vendored/forked submodule's local state is "just pre-existing, unrelated" during a future
  restructuring, `cd` into it and run `git status`/`git diff` directly** — a submodule flagged dirty
  in the parent's own `git status` may be carrying real, load-bearing, never-upstreamed work.

Each fork's own remotes: `origin` = the fork (push access), `upstream` = the official repo (`git
fetch upstream` to pull in future official releases without losing the local branch). Each fork's
own `.gitmodules` entry in its *parent* repo (`Experimental-Tasks/pybpod/.gitmodules` for the
rotaryencoder fork) pins both the fork URL and the `save-local-changes` branch explicitly, so a
fresh `git submodule update --init --recursive` anywhere in the chain reliably resolves to the
fork with the local changes, not a stock upstream clone missing them.

**Editable installs must be re-pointed after any of this** — `utils/install.py`'s `pip install -e`
step links Python's site-packages to whatever directory it was run against; cloning/updating a
submodule alone does *not* automatically repoint an *already-installed* editable package. Confirmed
directly: after the migration, `pybpod_rotaryencoder_module.__file__` still resolved to the OLD,
pre-migration `pybpod` checkout on disk until `pip install -e` was re-run from inside the NEW
`Experimental-Tasks/pybpod/...` location for every package in `utils/install.py`'s
`SUBMODULES_FOLDERS` list.

## Repository layout

- `pybpod/base/` — the three core packages that make up PyBpod itself:
  - `pybpod` (`pybpod/base/pybpod/setup.py`) — the top-level PyPI package and `start-pybpod`
    console-script entry point (`pybpodgui_plugin.__main__:start`).
  - `pybpod-api` — low-level Python API / driver for talking to the Bpod device over serial.
  - `pybpod-gui-api` — GUI-facing API layer between `pybpod-api` and the GUI plugin.
  - `pybpod-gui-plugin` — the main PyForms-based GUI application (Setups, Subjects, Boards, running
    protocols).
- `pybpod/libraries/` — shared/forked dependencies vendored as submodules rather than plain pip
  deps: `pyforms-gui`, `pyforms-generic-editor`, `logging-bootstrap`, `safe-collaborative-architecture`.
- `pybpod/plugins/` — optional GUI plugins that extend `pybpod-gui-plugin` via the generic-editor
  plugin system (each is independently versioned and pip-installable): session history, timeline,
  trial-timeline, stmdiagram, emulator, waveplayer, soundcard, **rotaryencoder** (forked, see
  above), **hifi** (from-scratch driver for the Sanworks HiFi HD audio module, lives in the
  `pybpod` fork directly, no separate upstream), alyx, and a terminal plugin (`pge-plugin-terminal`).
- `pybpod/docs/` — Sphinx documentation source, published to Read the Docs
  (https://pybpod.readthedocs.io/) — describes upstream PyBpod, not this lab's own task code below.
- `pybpod/utils/` — dev/release tooling (see Commands below).
- `_projects/` — this lab's own task code and training data, see below.

Because every `pybpod/` submodule is its own package with its own version pin, cross-cutting changes
(e.g. a new feature touching both `pybpod-api` and `pybpod-gui-plugin`) require editing multiple
submodules and bumping their versions independently — check `pybpod/CHANGELOG.rst` for the
shared-versioning history (PyBpod's own version has been kept in sync with `pybpod-api`,
`pybpod-gui-api`, and `pybpod-gui-plugin` since v1.8.0).

## Local project data (`_projects/`)

`_projects/` is a **real, mostly-committed subfolder of this repo** (not an external sibling
directory the way it was before the `Experimental-Tasks` migration). Each subfolder is one PyBpod
GUI project — e.g. `_projects/AuditoryEvidenceAccum/` — containing a `<ProjectName>.json` descriptor
plus `boards/`, `experiments/`, `subjects/`, `tasks/`, and `users/` folders that the GUI reads/writes
when a project is open.

**Only actual code (and one specific data file) gets committed here — everything else is
gitignored, deliberately and strictly**, per this repo's own `.gitignore`:
- `tasks/` (the real task scripts) — committed, this is source code.
- `_projects/*/records/training_log.xlsx` — committed, the **one explicit exception** to
  "no non-code files." See "Multi-box training-log sync" below for why this one file is special.
- `boards/`, `subjects/`, `users/`, `experiments/` (real session data, hardware/COM-port config) —
  gitignored. Board/subject configs are box-specific (different COM ports per machine); session
  data is exactly what `training_log.xlsx` summarizes, no need to duplicate the raw CSVs in git.
- `docs_local/` (design docs like `training_protocol.md`, bench-test docs like
  `HARDWARE_TEST_EXAMPLES.md`) — gitignored. These exist locally on this machine but are
  deliberately not shared via git; treat them as this machine's own reference material, not
  something every box needs a copy of.
- Any generated, non-`training_log.xlsx` artifact (`session_startup_checklist.xlsx`/`.md`, `.png`
  validation plots, `*_struct.mat`/`.json` session exports) — gitignored, regenerable/local-only by
  design.

Don't assume something is missing from a fresh clone just because it's not in git — `docs_local/`,
real session data, and machine-specific board/subject config all need to exist locally on each box
independently (the GUI creates/writes most of it automatically; design docs need a manual copy).

## Commands

Development environment (Python 3.6, conda-based), from this repo's own top level:

```bash
conda create -n pybpod-environment python=3.6
activate pybpod-environment   # Windows; use `conda activate` on macOS/Linux
git submodule update --init --recursive   # pulls in pybpod/ AND its own ~17 nested submodules
cd pybpod && python utils/install.py      # pip install -e each submodule, then generate user_settings.py
cd ..
start-pybpod                              # run the application
```

Confirmed working on this machine's `pybpod-environment` conda env (Python 3.6.13):`start-pybpod`
launches the Qt GUI and stays resident in its event loop. A one-line `WARNING ... Plugins path was
not defined by user` on startup is expected/benign, not a failure.

`pybpod/utils/install.py` does two things, run from *inside* `pybpod/`:
1. `pip install -e` every folder listed in `SUBMODULES_FOLDERS` (all of `base/`, `libraries/`,
   `plugins/`, resolved relative to `pybpod/`) so the whole stack is editable in-place. **Re-run
   this (or at least the `pip install -e` loop) any time a submodule gets re-cloned/updated** — an
   existing editable install does not automatically follow a submodule to a new location; see
   "Forked submodules" above. `base/pybpod` itself — the package whose `setup.py` actually defines
   the `start-pybpod` console-script entry point — is deliberately listed **last** in
   `SUBMODULES_FOLDERS`, after every other package it depends on (it pins exact versions of
   `pyforms-gui`, `pybpod-api`/`pybpod-gui-api`/`pybpod-gui-plugin`, and every `plugins/` package):
   installing it earlier would make `pip` try to satisfy those pins from PyPI instead of the local
   editable installs, since they wouldn't exist yet at that point in the loop. Confirmed as a real
   gap during a from-scratch machine setup: `base/pybpod` was missing from this list entirely for a
   time, so `python utils/install.py` alone left `start-pybpod` unregistered (`pip show pybpod` /
   `where start-pybpod` both came up empty) even though every other submodule installed cleanly —
   needed a separate `pip install -e base/pybpod` run last to fix it.
2. Writes `user_settings.py` (if it doesn't already exist, inside `pybpod/`) with
   `GENERIC_EDITOR_PLUGINS_LIST` set to the default plugin set (`pybpodgui_plugin`,
   `pybpodgui_plugin_timeline`, `pybpodgui_plugin_session_history`). **This file is gitignored
   (machine-specific) and does NOT travel with the repo** — a fresh box needs its own copy,
   customized with whatever plugins that box actually needs (e.g. `pybpod_rotaryencoder_module`,
   `pybpodgui_plugin_trial_timeline`, and `PYBPOD_API_MODULES = ['pybpod_hifi_module',
   'pybpod_rotaryencoder_module']` for the modules this lab's tasks actually use) — copying it from
   another already-set-up box is the fastest way to get a new one going. Confirmed as a real
   failure mode: a box with only the *default* `user_settings.py` silently loses the "Wheel
   Position" live-plot feature (and any other non-default plugin) with no error, since the GUI just
   never loads a plugin package that isn't in `GENERIC_EDITOR_PLUGINS_LIST`. **A second, related
   failure mode, confirmed directly**: `start-pybpod` resolves `user_settings.py` off
   `sys.path.insert(0, os.getcwd())` — i.e. off whatever directory it happens to be launched
   *from*, not a fixed repo-relative path. `pybpodgui_plugin/__main__.py`'s own fallback, on
   `ModuleNotFoundError`, silently creates (if one doesn't already exist) `~/user_settings.py`
   (the OS user's home directory) with only `GENERIC_EDITOR_PLUGINS_LIST = ['pybpodgui_plugin']` —
   no rotaryencoder, no timeline, no session-history, no trial-timeline — and uses it with **no
   error or warning of any kind**. Confirmed as a real, already-hit failure mode on this machine:
   launching `start-pybpod` from a PowerShell `cd`'d somewhere other than `pybpod/` silently picked
   up a genuinely ancient (dated 2020, unrelated prior use of this machine) minimal home
   `user_settings.py`, which manifested as every session's right-click context menu showing *only*
   "Remove" — none of the timeline/session-history/rotaryencoder/trial-timeline menu items, since
   none of those plugins were in that stale file's list at all (traced all the way down through
   `pyforms_generic_editor`'s plugin-mixin composition mechanism before finding the real cause was
   this cwd-dependent fallback, not a code bug — the composed `Session` class is provably correct
   when the right `user_settings.py` is actually in use). Fixed by overwriting the stale home file
   to match `pybpod/user_settings.py` exactly, so it now works correctly as a fallback regardless of
   launch directory — but the underlying cwd-dependency itself is unchanged, so **always `cd` into
   `pybpod/` before running `start-pybpod`** rather than relying on this fallback catching every
   case; a *different* home directory (a different Windows user account, a different machine) would
   still hit the same silent-wrong-plugin-list trap unless its own `~/user_settings.py` is checked.

There are OS-specific conda environment files under `pybpod/utils/` (`environment-windows-10.yml`,
`environment-ubuntu-17.10.yml`, `environment-macOSx.yml`) with pinned native deps (Qt, PyQt, HDF5,
etc.) — use the one matching the dev machine's OS instead of a fresh `conda create` if native builds
are an issue.

There is no build/lint/test tooling at the top level of either `Experimental-Tasks` or `pybpod`
itself (no top-level `pytest.ini`, `tox.ini`, or CI test workflow). `_projects/`'s own offline test
suite is `_projects/AuditoryEvidenceAccum/tasks/_wheel_shaping_shared/validate_wheel_shaping.py` —
run it after any change to `staircase.py`/`direction_tracker.py`/`session_csv_parser.py`/
`build_training_log.py`'s merging logic (see "Wheel-shaping training stages" below); it needs no
hardware.

### Versioning / release

- `pybpod/base/pybpod/.bumpversion.cfg` drives version bumps for the top-level `pybpod` package via
  `bump2version`/`bumpversion`, run from `pybpod/base/pybpod/`. It keeps `setup.py`,
  `../../docs/source/conf.py`, and `../../README.md` in sync.
- `pybpod/utils/deploy-pypi.py` walks `libraries/`, `base/`, and `plugins/`, compares each
  submodule's local `setup.py --version` against the latest release on PyPI, and runs `sdist
  bdist_wheel` + `twine upload` for any that are ahead. This is a real publish action — do not run
  it without the user's explicit intent to release.

## Hardware module notes

Custom Bpod hardware modules (e.g. `pybpod-gui-plugin-soundcard`, `pybpod-gui-plugin-hifi`,
`pybpod-gui-plugin-rotaryencoder`, all under `pybpod/plugins/`) follow a consistent two-file
pattern: `module_api.py` (direct USB connection to the module for hardware-facing calls) +
`module.py` (a `BpodModule` subclass — the state-machine-relay side that self-announces to Bpod,
e.g. `HiFi1`, `RotaryEncoder1`). `pybpod/plugins/pybpod-gui-plugin-hifi/` (package
`pybpod_hifi_module`) is a from-scratch driver added this way for the Sanworks HiFi HD audio module
(no prior Python support anywhere); it lives directly in this lab's own `pybpod` fork, not a
separate upstream submodule like the other `plugins/` entries — a deliberate open decision, not an
oversight.

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
  reliably responds to a handshake (`pybpod/plugins/pybpod-gui-plugin-rotaryencoder/
  pybpod_rotaryencoder_module/module_api.py`'s `BOOT_DELAY_S`/`HANDSHAKE_RETRIES`).
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
  `pybpod-gui-plugin-rotaryencoder`'s `wheel_position_plot.py` — now lives in that plugin's own
  fork, see "Forked submodules" above — added as a "Wheel Position" right-click option on session
  tree nodes, mirroring `pybpodgui_plugin_trial_timeline`'s
  session-treenode files). Slice only new rows (`iloc[rows_read:]`) instead of re-querying
  everything, and use persistent, updatable plot artists instead of clear-and-replot every tick —
  same "only touch what changed" principle on both the data and rendering side.

## Poisson-clicks / wheel-turn evidence-accumulation task

Building toward a wheel-turn analog of the Brunton-style Poisson-clicks 2AFC task — see
`_projects/docs_local/design_docs/mouse_auditory_accumulation_paradigm.md` (gitignored, see "Local
project data" above) for the original design and its `## 13. Implementation amendments` section for
how the built version deliberately diverges from it
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
- **A rig with multiple physical monitors for the dot stimulus may still only expose ONE combined
  Qt screen, not one per physical panel.** Confirmed on this rig via a direct, read-only
  `QApplication().screens()` query: three physical dot-stimulus monitors reported as a single
  6144x1536 `QScreen` (`DISPLAY2`; 6144 / 3 = 2048px, a standard panel width — almost certainly the
  GPU bonding the three outputs into one combined surface), alongside one separate, genuinely
  distinct 1920x1080 `DISPLAY1`. `DotDisplay` (unmodified) draws relative to that whole combined
  window's own center/width, so its full range of motion visibly crosses from the intended middle
  physical monitor onto the two outer ones as the wheel turns — there is no separate `QScreen` for
  those outer panels to black out independently. `_shared/dot_display.py`'s new, purely additive
  `MiddleScreenDotDisplay` (+ its own `_MiddleOnlyDotWidget`) fixes this correctly for that
  situation: it still spans the one combined window, but always paints every column except one
  `active_segment_index`-th of `n_segments` equal columns solid black, and confines the dot's
  rendering AND its full range of motion (via `get_screen_width_px()` returning just that active
  column's width) to that one column — a drop-in replacement for `DotDisplay` (identical public
  method surface), used by `dot_wheel_midscreen_test.py` and, behind a `VAR_USE_MIDDLE_SCREEN_ONLY`
  flag, `hifi_singleside_dot_test.py`. Before assuming "N physical monitors = N `QScreen`s" on any
  rig, check `screens()` directly rather than trusting the monitor count alone.
- **This rig's rotary encoder's raw positive-position direction is the physical direction that
  visually moves a wheel-coupled display element toward screen-LEFT, not screen-right** — confirmed
  directly on hardware (a user turning the wheel observed the dot moving opposite to the expected
  direction). This does NOT affect which physical direction crosses
  `VAR_LEFT_THRESHOLD_DEG`/`VAR_RIGHT_THRESHOLD_DEG` or which event (`left_event`/`right_event`)
  fires — those are driven directly by the rotary's own raw position sign via native firmware
  thresholds, entirely independent of anything rendered on screen. Centralized as
  `rotary_setup.WHEEL_TO_SCREEN_SIGN` (currently `-1`) and `rotary_setup.screen_direction_gain
  (magnitude)` — every wheel-coupled dot-display script (`dot_wheel_test.py`,
  `dot_wheel_midscreen_test.py`, `hifi_singleside_dot_test.py`, `full_protocol_lookback_test.py`)
  calls `rotary_setup.screen_direction_gain(...)` around its own gain calculation instead of
  hardcoding a sign flip locally, so a future rewiring/remounting only needs one constant changed,
  not every display-coupling script individually. **Not yet ported to the Gabor-based scripts**
  (`gabor_wheel_test.py` and its hifi-combined variants) — no complaint has been raised there, and
  they haven't been checked; don't assume they share (or don't share) this same sign issue without
  checking directly.
- **`full_protocol_lookback_test.py` has the same `VAR_USE_MIDDLE_SCREEN_ONLY` flag as
  `hifi_singleside_dot_test.py`** (full/spanned `DotDisplay` vs. `MiddleScreenDotDisplay`) — it was
  already `DotDisplay`-based (a prior pass had already swapped it in from `GaborDisplay`) by the
  time the sign-fix/screen-flag work happened, so both fixes just needed applying there too, no new
  display logic.
- **Camera recording (`_shared/camera_recorder.py`, `CameraRecorder`/`discover_camera()`)**: a USB
  camera recorded alongside a Bpod session, time-synced to the exact same `log_python_t0`-relative
  clock every other VAL registration in this codebase already uses (per-frame, not an assumed fixed
  fps — no live camera can guarantee an exact delivered rate, so a companion
  `<output>_timestamps.csv` records each frame's real `time.time() - log_python_t0` instead).
  Recording runs on its own background thread, fully decoupled from any caller's render-loop
  cadence — "nothing has to be done with the video during the task" was the explicit requirement,
  so it's fire-and-forget by design. Mirrors `rotary_setup.py`/`hifi_setup.py`'s house style and
  `bpod_trial_helpers.py`'s `TrialRunner` threading idiom exactly (one lock scoped to just the
  hardware access, a fresh `Event()` per `start()`, daemon thread, bounded `join()`, broad
  `except Exception: break`). Live preview reuses `QApplication.instance()` (same single-Qt-toolkit
  convention as `dot_display.py`/`gabor_display.py`) rather than `cv2`'s own `imshow()`/HighGUI
  window, specifically to avoid the same native-toolkit-conflict crash class already documented
  below (`TkAgg` + PyQt5). **Confirmed on hardware: the capture loop needs an explicit rate limit**
  — the first real run had no throttle at all (an oversight relative to `TrialRunner`'s own poll
  loop, which does sleep between iterations) and caused a real, noticeable system-wide lag, since
  `cv2.VideoCapture.read()` doesn't reliably block on every camera/backend combination; fixed with
  a "measure elapsed, sleep the remainder" throttle to `~fps`.
- **Two preview modes**: `.show()` for continuous repaint (used by the standalone
  `_projects/Tests/tasks/camera_test/camera_test.py` bench task — simple script, no competing
  background threads, continuous preview is fine there), and `.show_snippet(duration_s)` for a
  brief windowed preview that auto-hides after the deadline (used by
  `full_protocol_lookback_test.py`, which already runs a `WHEEL_POS` poll thread + a
  `decision_thread` and has the documented `PyEval_RestoreThread` GIL-pressure crash history below
  — a continuous preview repaint there would add to that same risk). `full_protocol_lookback_test.py`
  calls `show_snippet()` at exactly two moments per trial (decision-period start, right after the
  choice resolves) rather than every render-loop iteration — confirmed as a real, worthwhile
  further lag reduction on top of the capture-loop throttle, not just a cosmetic choice, since the
  per-frame `cv2.cvtColor`/`QImage`/`QPixmap` conversion+repaint is the dominant remaining
  main-thread cost once the background thread itself is rate-limited. The full video recording is
  completely unaffected by which preview mode (or neither) is active.

## Wheel-shaping training stages (Stage 1/2, `AuditoryEvidenceAccum`)

`training_protocol.md` (in `_projects/docs_local/design_docs/`, alongside
`mouse_auditory_accumulation_paradigm.md` — gitignored, local-machine-only, see "Local project
data" above) is the full staged training curriculum (Part 4: Stage 1 through Stage 8) for the
wheel-turn/dot variant of the evidence-accumulation task. Stage 1 ("Wheel unlocked, spout close, small movements rewarded")
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
  automatically by the task script itself, no manual data entry between sessions. **`VAR_SUBJECT_ID`
  is derived automatically from the GUI's own selected subject**, not hand-edited — both stage
  scripts do `from confapp import conf as settings; ...
  session_csv_parser.parse_subject_name(settings.PYBPOD_SUBJECTS[0])` (confirmed this resolves
  correctly: `pybpodapi/__init__.py` itself does `import user_settings; conf += user_settings` at
  import time, picking up the session-local `user_settings.py` `board_com.py` generates per run —
  same settings-resolution mechanism already documented below for `PYBPOD_API_MODULES`; a script
  reading `settings.PYBPOD_SUBJECTS` only needs to import `pybpodapi` *before* reading it, which
  both scripts already do via `from pybpodapi.protocol import Bpod, StateMachine`). Raises loudly
  (`RuntimeError`) if `PYBPOD_SUBJECTS` is empty rather than silently falling back to a placeholder
  — a prior hardcoded `VAR_SUBJECT_ID = 'REPLACE_ME'` constant was confirmed, on real hardware, to
  have never been updated across many real sessions, silently pooling all of one animal's real
  staircase progress under a shared placeholder identity instead of its own. **This is a
  DIFFERENT identity from the GUI's own project-level "subject" used for `SUBJECT-NAME`/session
  merging** (see `build_training_log.py` below) — both now derive from the same underlying GUI
  selection, but conceptually serve different consumers (persisted staircase state vs. the training
  log), so don't assume changing one automatically explains behavior tied to the other. **Stage 1
  and Stage 2 share one continuous state file per subject** (same filename, same path derivation) —
  Stage 2's own new keys (`threshold_fraction` etc.) default to Stage 1's known ending values the
  first time it runs for a given subject, rather than each stage starting its own separate record.
- **Stage 1's wheel gain (`VAR_GAIN_INITIAL_MULT`) decays by a FIXED step per qualifying session
  (`staircase.GAIN_DECAY_STEP_PER_SESSION = 0.1`), floored at `staircase.GAIN_FLOOR_MULT = 2.0`** —
  same `grow_iti()`-shaped mechanism as ITI/Stage-1-threshold growth, not the multiplicative decay
  (`x0.90`/session toward a `1.0x` floor) an earlier version used. Confirmed on hardware the
  multiplicative version's `1.0x` floor made movements barely discernible late in Stage 1; `3.0x` ->
  `2.0x` over ~10 qualifying sessions was chosen instead. `decay_gain()` in `staircase.py` is the
  one function to change if this rate/floor needs retuning again — same call site
  (`state.set('gain_mult', staircase.decay_gain(cur_gain_mult))`), only the internal math differs
  from `grow_iti()`'s (subtracting instead of adding).
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
- **`DirectionRatioTracker.right_fraction()` needs a `has_full_window()` gate before its ratio is
  trusted for withholding decisions** — same class of bug as the `should_stop_session()` rate-check
  gotcha already documented above, just not originally applied here too. Confirmed directly in a
  real Stage 2 session log: `DIRECTION_RATIO=0.0` was already registered after trial 1 (a single
  `'L'` trial pushes the raw ratio to an extreme with zero real signal behind it), and
  `OUTCOME=Withheld` fired as early as trial 2. `in_band()` now returns `True` unconditionally until
  `len(self._sides) >= self._sides.maxlen` (the full ~40-trial window) — `right_fraction()` itself
  stays unchanged (still reports the real rolling value for logging/display regardless of window
  fullness), only the withhold-triggering `in_band()`/`over_used_side()`/`should_withhold()` chain
  is gated.
- **`wheel_position_plot.py`'s live "Wheel Position" plot only ever recognized `LEFT_THRESHOLD_DEG`/
  `RIGHT_THRESHOLD_DEG` VAL names for its threshold reference lines** — neither stage script has
  ever registered those; both register a single symmetric `THRESHOLD_DEG` instead (matching
  `rotary_setup.set_and_enable_thresholds(rotary, [-cur_threshold_deg, cur_threshold_deg])`'s own
  symmetric convention), so the threshold lines never appeared. Fixed by also recognizing
  `THRESHOLD_DEG` and drawing both `±value` lines from it (LEFT/RIGHT handling kept, in case a
  future task genuinely has asymmetric thresholds). This lives in the rotaryencoder fork (see
  "Forked submodules" above) — any fix here needs the same commit-to-both-checkouts-then-push
  discipline, not just an edit to the working copy.
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
- **`AuditoryEvidenceAccum/records/`** holds the lab's record-keeping — see "Multi-box training-log
  sync" below for the full design (`build_training_log.py`, session merging, struct export, the
  `.bat` launchers) — plus `session_startup_checklist.xlsx`/`.md` (a static, printable form template
  generated once by `build_startup_checklist.py`, both formats from one shared `CHECKLIST_SECTIONS`
  data structure — animal ID, protocol run, equipment on/off checklist, pre-session checks, outcome
  summary; purely physical record-keeping, never auto-populated, since none of it has a digital
  source).

## Multi-box training-log sync (`AuditoryEvidenceAccum/records/build_training_log.py`)

`training_log.xlsx` is the **one non-code file this repo commits** (see "Local project data"
above) — one workbook, one sheet per animal, shared across every behavior box via plain `git
pull`/`push`, each box only ever able to see its own local session CSVs. This drove several
non-obvious design choices:

- **Update-in-place, not rebuild-from-scratch.** Rebuilding fresh every run (an earlier design)
  would silently wipe every OTHER box's sheet it can't see locally the moment two boxes' xlsx
  copies diverged even slightly. `build_workbook()` instead loads the existing file if present and
  touches ONLY the sheet(s) for subjects it found sessions for locally — every other sheet is left
  byte-for-byte untouched. There are no more separate `animals_metadata.json`/
  `session_manual_entries.csv` files (an earlier design) — DOB/Sex/Strain/Baseline weight
  (hand-typed once, in each sheet's own header block) and per-session `Weight (g)`/`Weight after
  task (g)`/`Notes`/`Water given by hand after task` (hand-typed per row) are now fields directly
  in the xlsx itself, since that's the one file already committed and synced; the script never
  overwrites a cell that's already been hand-filled.
- **`Weight after task (g)`/`Weight after task %`/`Water given by hand after task`** — three
  columns added alongside the original `Weight (g)`/`Weight %` pair, same placement logic
  (clustered together, right after `Weight %`, pushing `Trials`/etc. down). `Weight after task (g)`
  and `Water given by hand after task` are **manual** (added to `_MANUAL_COLUMNS`, carried forward
  across each run's full-sheet rewrite via the same read-before-clear/carry-forward-by-session-
  identity mechanism `Weight (g)`/`Notes` already use — extended the same non-generalized way,
  two more explicit keys in the same two spots, not refactored into a loop). `Weight after task %`
  is a **formula column**, byte-identical pattern to `Weight %`
  (`=IFERROR(<weight cell>/$<baseline cell>,"")`, refreshed every run) just referencing
  `Weight after task (g)`'s own resolved column letter instead of `Weight (g)`'s — both formulas
  read the *same* per-subject baseline weight cell, so a session's before/after weight both stay
  comparable to the same reference point.
- **`scan_all_sessions()` now scans the `Tests` project too, not just `AuditoryEvidenceAccum`**
  (`_SCAN_PROJECT_DIRS`) — added per explicit request once `full_protocol_lookback_test.py` (see
  "Poisson-clicks..." above) was confirmed working well on hardware, so its sessions could start
  feeding into the same per-animal training log. **Only protocols with a real `PROTOCOL_CONFIG`
  entry generate a row at all** (`summarize_session()` now returns `None` for anything else) — the
  `Tests` project has many bench-test protocols (`dot_wheel_test`, `camera_test`, `lick_reward`,
  ...) with no meaningful outcome taxonomy to report, and blindly including all of them would flood
  the log with all-`'unknown'`-outcome noise. Currently only `full_protocol_lookback_test` has a
  config entry beyond stage1/2. Its rows land in their own **`Test` section**
  (`_stage_label()`/`_TEST_PROTOCOL_LABELS`) within the relevant subject's own sheet, sorted after
  every numbered stage (the existing `_stage_sort_key()` fallback already sorted non-`stageN_`
  protocols last, alphabetically — no change needed there).
- **`full_protocol_lookback_test` needed two genuinely new outcome types in `classify_trial()`,
  not a fit into Stage 1/2's existing vocabulary.** It's a real 2AFC choice protocol — `incorrect`
  (`ErrorConsumption` visited, a real wrong-choice outcome) and `aborted` (`WheelAbort` visited, a
  mid-*cue*-period wheel movement from the separate `cue_sma` that runs before the choice state
  machine even exists for that trial — never co-occurs with rewarded/incorrect/no-response, since
  an aborted trial never reaches the second state machine at all) have no equivalent in Stage 1/2's
  shaping-only outcome set (`rewarded`/`withheld`/`no_movement`). Approximating them onto
  `withheld`/`aborts` was considered and rejected — those columns would then mean different things
  for different protocols' rows in the same sheet, which is exactly the kind of silent mismapping
  this codebase's own "native-STATE/EVENT-parsing" preference exists to avoid. Added as real new
  columns instead: `Incorrect`, `Wheel Aborts` (both auto, summed across a merged group same as
  `Withheld`/`No-movement`) — distinct from the pre-existing `Aborts` column, which is Stage 1/2's
  own `QUIESCENCE_BREAKS` VAL count and means something different again. `no_movement_states` is
  reused for `NoResponse` (a response-timeout, not "never moved the wheel") — close enough in
  spirit ("no choice made") that a new column wasn't worth it there. `side_from: 'event'` for this
  protocol, keyed on the same raw `RotaryEncoder1_3`/`RotaryEncoder1_4` threshold-crossing events
  the task script itself uses for `left_event`/`right_event` (its `Reward`/`ErrorConsumption` state
  names don't carry an L/R suffix the way Stage 2's `RewardL`/`RewardR` do, so `side_from: 'state'`
  doesn't apply here — same reasoning Stage 1 already established for its own `side_from: 'event'`
  choice).
- **Column layout is fully canonicalized (header row included) on every run**, not just
  append-only. An earlier design only ever *appended* new columns to whatever historical order a
  sheet already had, self-migrating the schema but never actually fixing a column's *position* once
  set — confirmed as a real, user-visible problem: `Weight (g)`/`Weight %` stayed stuck in their
  original positions across several schema revisions despite `COLUMNS`'s own list order changing
  each time. `_rewrite_subject_sheet()` now fully clears the table (`ws.delete_rows()`, header row
  included) and rewrites every column fresh from `COLUMNS`' own fixed order every run — a column
  reorder in code now actually reorders the real sheet, not just future ones. Old, now-orphaned
  labels (e.g. a since-removed `Protocol / Stage` column — redundant once rows are grouped into
  per-stage sections with their own header row, so it was dropped entirely) simply don't get
  recreated; nothing of value is lost since retired columns were already fully superseded.
- **`ws.column_dimensions` (width/hidden) is keyed by LETTER and survives `delete_rows()`/schema
  changes** — a real gotcha, confirmed directly: the hidden `Session started` key column used to sit
  at a different letter in an earlier schema iteration, and that letter's `hidden=True` flag
  lingered forever after the column moved elsewhere, making an unrelated *visible* column (`Num.
  Sessions`) appear hidden too. Every column's `hidden` flag is now explicitly reset to `False`
  before re-hiding only whichever letter `Session started` currently resolves to, every run — don't
  assume a fresh `_canonical_col_map()` write alone clears stale column-level (as opposed to
  cell-level) formatting.
- **Session merging (restart-combining)**: a session that got Stopped, Killed, or crashed partway
  through and was restarted within an hour (`_MERGE_GAP = 3600.0`s, same protocol only) is combined
  into ONE row — see `group_sessions()`/`combine_group()`. A session that reached its own natural
  end (`VAR_MAX_TRIALS`, registered as `SESSION_END_REASON='completed'` in the task script's own
  `for...else` branch — nothing is registered on the Stop/Kill/crash paths, so *absence* uniformly
  covers all three) always ends a group, never merging with what follows. Applies to every subject
  uniformly, including bench-test ones — an earlier design specifically excluded the `test` GUI
  subject from merging, which was reverted per explicit instruction once it became a recurring
  point of confusion ("why didn't these two sessions merge" turned out, twice, to be "they were run
  under the `test` GUI subject" before the exclusion was removed).
  - A session's own end time comes from the native `SESSION-ENDED` INFO row (`Bpod.close()` writes
    it) via `session_csv_parser.session_end_datetime()`; falls back to `started + event-span
    duration` when absent (the crash case — nothing wrote a clean end record).
  - Combining a group sums per-trial tallies (trial count, L/R turns, licks, aborts, consumed
    reward volume) and each member's own event-span `session_duration_s` (**not** the wall-clock
    gap between them — confirmed as a real bug and fixed: naively using `last.ended_dt -
    first.started_dt` inflated a plain, unmerged singleton session's own Duration from ~8s to
    ~116s by including Bpod connection/handshake overhead between `SESSION-STARTED` and the first
    real trial). `threshold_start_deg` comes from the first member, everything else "as of end of
    session" (`threshold_end_deg`, `gain_mult_end`, `direction_ratio_end`, `reward_ul`) from the
    last — same semantics a single session's own `_end` fields always had, just now "end of bout."
  - `Num. Sessions` is blank for an ordinary un-merged row (not `1`) — a bare `1` there read as if
    something had been combined when nothing had; it only shows a count once a real merge happened.
  - The hidden `Session started` column stores ALL member session-start timestamps, `; `-joined —
    this is the actual matching key across runs (exact match, or a strict-subset match when a
    restart later absorbs a session that used to stand alone into a bigger group), which is why
    hand-typed `Weight (g)`/`Notes` values correctly carry forward across a merge appearing later.
    **Splitting an already-merged group back apart (e.g. temporarily testing/reverting a merge
    behavior) is the one case this carry-forward doesn't handle** — the old row's superset key
    doesn't subset-match either resulting singleton, so hand-typed values on that row get orphaned
    and need restoring by hand. Confirmed hit twice while iterating on the merge-exclusion
    behavior above; not expected to come up in normal forward-only usage (real restarts only ever
    grow a group, never shrink one).
- **`session_struct_export.py`** (`_wheel_shaping_shared/`) writes a `.mat` (`scipy.io.savemat`,
  MATLAB-openable, `trials` comes out as a 1xN cell array of structs — the same shape as Bpod's own
  native `SessionData.RawEvents.Trial`) and `.json` (Python-readable) snapshot of a session,
  wired into both stage scripts at every termination path (normal finish, Stop, Kill) — enough to
  reconstruct exactly what a task was configured to do (`task_params`, harvested automatically via
  `{k: v for k, v in globals().items() if k.startswith('VAR_')}`, not a hand-maintained list) plus
  everything that happened. **The session's own CSV path must be captured BEFORE calling
  `my_bpod.close()`, not after** — confirmed directly in `bpod_base.py`: `close()` does `del
  self._session`, so `my_bpod.session._path` raises/is gone once `close()` has run. Local-only,
  never committed (see "Local project data" above). `merge_session_structs()` combines multiple
  raw sessions' own already-exported structs (preferring each member's own `.json`, which has
  `task_params` — not recoverable by re-parsing the raw CSV alone, since `VAR_*` values are never
  written into the CSV itself) into one `<first_session>_combined_struct.mat`/`.json`, called from
  `build_training_log.py`'s grouping step for any merged group.
- **One native Bpod INFO row's own "key" is a full descriptive sentence** ("This is a PYBPOD file.
  Find more info at ..."), not a short identifier like every other INFO key — confirmed present in
  every real session CSV's preamble, not a parsing bug. Broke `scipy.io.savemat` (`ValueError:
  Field names are restricted to 63 characters`) the first time struct export tried to use it
  directly as a MATLAB struct field name; `session_struct_export._sanitize_key()` strips
  non-identifier characters and truncates to 63 chars for exactly this reason — worth remembering
  for any future consumer that turns raw INFO-row keys into identifiers of any kind.
- **Batch launchers** in `records/`, for anyone who'd rather double-click than use a terminal:
  `Update Training Log.bat` (local refresh only, no git action), `Sync Training Log.bat` (`git
  pull` -> refresh -> commit+push `training_log.xlsx`, but only if it actually changed — a failed
  `pull` from a real conflict, which *will* happen eventually since `.xlsx` is a binary format git
  can't auto-merge, stops the script immediately with a clear message rather than risking a silent
  overwrite), and `Print Startup Checklist.bat` (regenerates the blank template). All three `cd
  /d "%~dp0"` first and try a hardcoded known-good Python path before falling back to `conda
  activate pybpod-environment`, so they work whether or not that exact interpreter path exists on
  a given box.

## Liquid reward (valve) calibration (`Calibration/`)

Water reward volume was never calibrated anywhere in this codebase until this section's own work —
`VAR_REWARD_UL` in the wheel-shaping stage scripts was a purely cosmetic placeholder (only ever
`register_value()`'d for logging, never fed into `VAR_REWARD_DURATION`, which is what actually
drives the valve, hardcoded to `0.1`s in every task script that fires one). `Calibration/` ports the
same underlying approach Sanworks' original MATLAB Bpod software uses
(`Bpod_Gen2/Functions/Calibration/Liquid Reward/LegacyBpodLiquidCalibration.m`/
`LegacyGetValveTimes.m` — the current MATLAB `master` has moved to a newer OOP wrapper around the
same idea; `pybpod-api` itself has no calibration functionality of its own at all, confirmed by
reading its source tree, so this is this lab's own from-scratch port, not something reused from the
Python Bpod stack):

- **Data collection**: fire a valve N times (100-500) at a given duration, weigh the *total*
  dispensed liquid in grams, convert to a per-pulse volume assuming water density 1 g/mL
  (`volume_uL = mass_g * 1000 / N`), append `(duration_ms, volume_uL)` to that valve's table.
  Repeat at a few different durations — needs ≥3 points before fitting (same minimum MATLAB's own
  `LegacyGetValveTimes.m` enforces).
- **Fit**: per-valve 2nd-order polynomial fit as `duration_ms = f(volume_uL)` (`numpy.polyfit(
  volumes, durations_ms, 2)`) — **not** the reverse, and not piecewise interpolation between raw
  points, matching MATLAB's own direction/order exactly.
- **Lookup**: `LiquidCalibration.get_valve_time_s(volume_ul, valve_id=1)` evaluates that polynomial
  and returns the open duration directly in **seconds**, ready to assign straight to a `state_timer`
  — the Python equivalent of `GetValveTimes(volume_uL, valveID)`. Raises `ValueError` rather than
  ever silently returning a guessed duration if that valve hasn't been fit yet.
- `Calibration/liquid_calibration.py` — the importable data/fit/lookup module (`LiquidCalibration`
  class), JSON-persisted (not MATLAB's `.mat` — git-diffable, same precedent as
  `wheel_shaping_state.json` elsewhere in this repo) at `Calibration/liquid_calibration.json`.
  **That JSON file is gitignored, not committed** — a valve's physical calibration (tubing, wear,
  reservoir height) is specific to *this* rig's own hardware, same reasoning already applied to
  `boards/`/`subjects/` configs being machine-specific and gitignored elsewhere in this repo. The
  `.py` files themselves ARE committed, as shared tooling.
- `Calibration/calibrate_liquid.py` — the interactive data-collection front end: a standalone PyQt5
  GUI (valve dropdown, editable duration/pulse-count/mass fields, a live points table, fit button,
  and a volume→duration lookup preview) — the Python/PyQt5 equivalent of MATLAB's own
  `BpodLiquidCalibration('Calibrate')` GUI, not a command-line prompt sequence. PyQt5 needed no new
  dependency (already pulled in by `pyforms-gui`). Deliberately **not** a PyBpod GUI task/plugin —
  calibration is a bench maintenance procedure, not project/session-tracked experiment data, so it
  connects to the Bpod board directly (`Bpod()`, same convention every task script uses) and skips
  all PyBpod project/task scaffolding. Firing the requested pulses (each pulse a real one-state Bpod
  `StateMachine`, hardware-timed — not a Python `time.sleep()` loop, same "prefer native Bpod timing"
  principle used throughout this codebase) runs on a background `QThread` so the UI stays responsive;
  the pulse-related buttons are disabled for the run's duration so only one thread ever touches the
  Bpod connection at a time (per the "background thread can never safely share a live Bpod
  connection with another thread" hazard documented under "Hardware module notes" below).
- **`calibrate_liquid.py` needs `Bpod(serial_port=VAR_BPOD_SERIAL_PORT)`, not a bare `Bpod()` —
  confirmed as a real, previously-shipped bug, not a hypothetical.** `BpodCOMProtocol.__init__`
  (`pybpod-api`) only calls `self.open()` automatically `if self.serial_port:`, and
  `self.serial_port` normally resolves from `settings.PYBPOD_SERIAL_PORT` — which every *real* task
  script gets for free from the session-local `user_settings.py` the PyBpod GUI auto-generates per
  run (confirmed on disk: `PYBPOD_SERIAL_PORT = 'COM10'` in dozens of session folders). This tool
  deliberately skips all of that GUI/project scaffolding, which also means it skips the one
  mechanism that gets a serial port into `Bpod()` automatically — so a bare `Bpod()` here silently
  never opens, `hardware.max_states` stays `None`, and "Run Pulses" fails with `TypeError: can't
  multiply sequence by non-int type 'NoneType'` deep inside `state_machine_base.py`. This shipped
  broken (the tool's very first hardware test hit exactly this, and it went unfixed and
  undocumented for a while — confirmed via git history: `Calibration/`'s entire history was one
  commit that added both files, no follow-up) until traced all the way through the actual
  `Bpod`/`BpodIO`/`BpodCOMProtocol` MRO chain. Fixed by adding `VAR_BPOD_SERIAL_PORT = 'COM10'` as
  a module-level, machine-specific constant (same "edit per box" convention as
  `VAR_ROTARY_USB_PORT`/etc. elsewhere) and passing it explicitly. **No real calibration data had
  ever been collected before this fix** (`liquid_calibration.json` didn't exist on this machine).
- **Fit-curve plot** (added per explicit follow-up — "similar to what is implemented in matlab"):
  a `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg` embedded directly in
  `CalibrationWindow`'s own PyQt5 layout (genuinely embedded, not a separate popup) — scatters the
  selected valve's raw points and overlays the fitted curve once one exists, refreshed from the
  same two places (`_refresh_table()`, `_on_fit()`) the points table/coefficients already refresh
  from, so it can't go stale relative to what else is on screen.
- **Not yet done, deliberately scoped out for now**: no existing task script has been wired to call
  `get_valve_time_s()` in place of its own hardcoded `VAR_REWARD_DURATION` — that's a ~12-file,
  two-project change, intentionally left as a separate follow-up once real measured calibration data
  exists to validate against, rather than bundled into building the tool itself.

## Architecture notes

- The GUI plugin system (`pybpod-gui-plugin` + everything in `pybpod/plugins/`) is built on the
  `pyforms-generic-editor` framework (in `pybpod/libraries/`), which loads plugins by Python package
  name from `GENERIC_EDITOR_PLUGINS_LIST` in `pybpod/user_settings.py`. Adding a new plugin
  submodule to the repo is not enough by itself — it must also be added to that list (and
  installed) to actually load. `SETTINGS_PRIORITY` in `user_settings.py` controls settings-file
  precedence when multiple `user_settings.py` files are discoverable (see
  `logging-bootstrap`/`pyforms` settings resolution).
- `pybpod-api` talks to the physical Bpod device over serial; `pybpod-gui-api` and
  `pybpod-gui-plugin` are the layers that expose that over the desktop GUI (built on `pyforms-gui`,
  a Qt-based forms framework). Hardware-facing changes typically belong in `pybpod-api`; UI/workflow
  changes belong in `pybpod-gui-plugin` or the relevant plugin under `pybpod/plugins/`.
- Because packages are versioned and pinned against each other (see
  `pybpod/base/pybpod/setup.py`'s `install_requires`), a change in one submodule generally isn't
  picked up by a pip-installed (non-editable) instance of another until versions are bumped and
  republished — this only doesn't matter in the `-e` editable-install dev setup produced by
  `pybpod/utils/install.py`.
