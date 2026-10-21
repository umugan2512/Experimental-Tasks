# Parameter progression across the wheel-shaping curriculum

How five training parameters — movement/response threshold, ITI, cue/stimulus duration,
cue-to-go delay, and hold-to-initiate (quiescence) — evolve within a session and carry across
sessions, through Stages 1-5 (`stage1_wheel_shaping` … `stage5_resume_staircases`).

All cross-session state lives in one JSON file per subject:
`AuditoryEvidenceAccum/subjects/<subject_id>/wheel_shaping_state.json`, read/written by
`session_state.StageState`. A key a later stage declares for the first time is seeded from its
own default; a key already on disk from an earlier stage carries forward untouched until a stage
that actually updates it runs. "Qualifying session" criteria differ by stage (see each section) —
Stage 1/2 count every completed trial, Stages 3/4/5 count only `main` trials with a genuine L/R
response (excluding warmup, aborts, and no-response trials).

## 1. Movement / response threshold

The wheel-turn amount (in degrees, as a fraction of `VAR_THRESHOLD_FINAL_DEG = 35`) required to
register a response.

| Stage | Within-session | Across-session |
|---|---|---|
| 1 | Fixed for the whole session. | Grows once per qualifying session (>200 trials): `+0.02` fraction, ceiling `0.20` (Stage 2's own starting point). Starts at `0.10` (3.5°). |
| 2 | Per-trial `ThresholdStaircase`: `+0.10` after 20 consecutive successes (success = cleared the threshold at all, independent of debiasing withhold), `-0.10` after 5 consecutive failures. Ceiling `1.0`, floor `0.0`. | Persists `threshold_fraction` + both streak counters every session, so a streak survives overnight. |
| 3 / 4 | Frozen — read once at session start, never updated. | Passthrough only; defaults to `1.0` (Stage 2's ceiling) if a subject somehow skipped Stage 2. |
| 5 | Resumes the *same* `ThresholdStaircase` mechanism as Stage 2, continuing from whatever Stage 2/4 left persisted — but only on sessions where Tag B's selector (below) picks `'response'`; otherwise frozen for that whole session. | Target for advancement is `0.90` (`STAGE5_RESPONSE_THRESHOLD_FINAL_FRACTION`), **not** the staircase's own `1.0` ceiling — the staircase can still step past 0.90, advancement just stops requiring it to. |

## 2. ITI (inter-trial interval)

The correct/no-response-path wait before the next trial. (Error and abort trials get their own
fixed windows — see "Punishment and error/abort ITIs" below.)

| Stage | Within-session | Across-session |
|---|---|---|
| 1 | Fixed `0.5s`. | Never changes. |
| 2 | Fixed for the whole session. | Grows once per qualifying session (>200 trials): `+0.1s`, ceiling `1.5s`. |
| 3 / 4 | Per-trial `BoundedStaircase`: `+0.25s` after 20 consecutive correct `main` trials, `-0.25s` after 5 consecutive incorrect. Floor `1.5s` (Stage 2's ceiling), ceiling `3.0s`. | Persists `iti_s` + both streak counters, continuous across Stage 3 → 4 (one ramp, not two). |
| 5 | **Reverts to a fixed constant, `VAR_ITI_S = 1.5s`.** Does not read or write `iti_s` at all. | See "Known quirks" — this discards whatever Stage 3/4 ramped `iti_s` up to. |

## 3. Cue / stimulus duration

The auditory click-train duration.

| Stage | Within-session | Across-session |
|---|---|---|
| 1 / 2 | No auditory stimulus. | — |
| 3 | Per-trial `BoundedStaircase`: `+0.25s` after 20 consecutive correct `main` trials, `-0.25s` after 5 consecutive incorrect. Floor/start `0.25s`, ceiling `2.0s`. | Persists `stim_duration_s` + both streak counters. |
| 4 | Continues the same staircase, uninterrupted from Stage 3 (one ramp, not two). | Same keys as Stage 3. |
| 5 | **Reverts to a fixed constant, `VAR_STIM_DURATION_S = 0.25s`.** Does not read or write `stim_duration_s` at all. | See "Known quirks" — discards Stage 3/4's ramp. Comment in code: "extended only from Stage 6 onward." |

## 4. Cue-to-go delay ("Tag A", `DelayPeriod`)

The gap between the cue ending and the response window opening.

| Stage | Within-session | Across-session |
|---|---|---|
| 1 / 2 | N/A — no cue, no delay period. | — |
| 3 / 4 / 5 | Fresh uniform random draw every trial, `[0.1s, 0.2s]`. **Not staircased at all.** | No persisted state — nothing to carry. |

Comment in code: "staircased 0-750ms only from Stage 7 onward, not built here." There is no
delay-staircase class anywhere in this codebase yet.

## 5. Hold-to-initiate (quiescence)

How long the wheel must sit still before a trial can start. `current_s` is the **scale
parameter of an exponential draw**, not a fixed duration — the actual per-trial hold requirement
is `clip(Exponential(current_s), floor, ceiling)`.

| Stage | Within-session | Across-session |
|---|---|---|
| 1 | Fixed `0.1s` (100ms). | Never changes. |
| 2 | Fixed `0.1s` **until** a one-way latch unlocks it: the first time, within any Stage 2 session, the movement-threshold staircase (§1) steps *up* at least once (checked every trial, not just start-vs-end of session — a session that steps up and later back down still counts). Once unlocked, that same session still runs on the fixed 0.1s; staircasing begins the **next** session. | Persists `quiescence_unlocked` (one-way, never reset), and once unlocked, persists `quiescence_s` + both streak counters. |
| 2 (post-unlock) / 3 / 4 | Per-trial `QuiescenceStaircase`: `+0.025s` after 20 consecutive clean holds (zero resets), `-0.025s` after 5 consecutive holds with at least one reset. Floor `0.1s`, ceiling `0.25s`. | Continuous across Stage 2 → 3 → 4 once unlocked — one ramp, not three. A subject that somehow skips Stage 2 entirely falls back to the fixed `0.1s` in Stage 3/4 rather than crashing. |
| 5 | Continues the *same* persisted staircase — not a fresh reintroduction. Only advances on sessions where Tag B's selector picks `'quiescence'`; frozen otherwise. | May already be at the `0.25s` ceiling before Stage 5 even starts, if Stage 2/3/4 already got there — expected, not a bug (see "Known quirks"). |

**Tag B (Stage 5 only)**: at most one of {movement/response threshold, quiescence} may advance
in any given Stage 5 session — `select_stage5_staircase_to_tighten()` picks whichever is
furthest (as a normalized 0-1 fraction) from its own final value, tie-breaking by alternating
away from the previous session's pick (persisted as `stage5_last_staircase_advanced`). The
*inactive* one is fully frozen that session — its counters don't move either.

## Punishment and error/abort ITIs (not staircased, any stage)

- **Wrong-direction wheel turn** (Stages 3/4/5 only — Stages 1/2 have no correct-side concept at
  all): `ErrorConsumption` (0.5s lick-detection window) → `ErrorPunish` (350ms, LED + full-white
  screen) → `IncorrectITI` (fixed `VAR_INCORRECT_ITI_S = 4.0s`).
- **Mid-cue wheel-movement abort** (Stages 3/4/5 only): `WheelAbort` → `AbortPunish` (350ms, LED
  + full-white screen) → `AbortITI` (uniform random `[1.0s, 2.0s]`).
- A plain timeout (`NoResponse`, no choice made at all) gets neither the punishment flash nor
  either of these — it proceeds straight to the normal ITI (§2).
- None of `VAR_INCORRECT_ITI_S`, the abort-timeout range, or the punishment-flash duration
  (`VAR_ERROR_FLASH_S = 0.35s`) are staircased or persisted — they're fixed per-stage constants.

## Known quirks (not fixed here)

These are real, confirmed behaviors of the current code — documented so they're not mistaken for
bugs during a future change, and flagged in case they're worth revisiting on their own:

- **Stage 5 discards Stage 3/4's ITI ramp**, reverting to a fixed 1.5s (§2).
- **Stage 5 discards Stage 3/4's cue-duration ramp**, reverting to a fixed 0.25s (§3).
- **`threshold_fraction`'s default value is inconsistent across stages**: Stage 3/4 default to
  `1.0` (Stage 2's ceiling) if a subject somehow skips Stage 2, while Stage 5 defaults to `0.45`
  (`0.90 * 0.5`) under the same circumstance — different fallback philosophies for the same key.
- **`click_attenuation_sessions_remaining` decrements once per session *run*, not per qualifying
  session** — unlike every other cross-session step in this curriculum. A crashed/aborted Stage 3
  session still burns one of its two warmup-attenuation sessions.
- **A Stage 5 session's `quiescence_at_final` gate may already be trivially true on its very
  first session** if Stage 2/3/4 already pushed `quiescence_s` to the `0.25s` ceiling before
  Stage 5 even started. This is the intended consequence of unlocking quiescence staircasing as
  early as Stage 2 rather than introducing it fresh at Stage 5 — in that case Stage 5's own job is
  just the response-threshold staircase.
- **Qualifying-session criteria are not uniform**: Stage 1/2 count every completed trial; Stage
  3/4/5 count only `main` trials with a genuine L/R response. "200 trials" means something
  materially different before vs. after Stage 3. The number of consecutive qualifying sessions
  required to advance also differs: 2 (Stage 1), human-judgment-call (Stage 2 — no automatic
  "advance-ready" check exists), 3 (Stage 3), 3 (Stage 4), 2 (Stage 5).
