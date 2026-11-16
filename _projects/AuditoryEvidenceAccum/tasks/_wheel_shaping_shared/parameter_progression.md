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
| 1 | Fixed for the whole session. | Grows once per qualifying session (≥100 rewarded trials; same-day and Stopped sessions count too): `+0.02` fraction, ceiling `0.20` (Stage 2's own starting point). Starts at `0.10` (3.5°). |
| 2 | Per-trial `ThresholdStaircase`: `+0.10` after 20 consecutive successes, `-0.10` after 5 consecutive failures. Success = a rewarded turn; a debiasing-withheld turn is a failure (since 2026-11-02, so an animal stuck on one side gets an easier threshold). A NoMovement trial is neutral since 2026-11-05 (`record_neutral()`): it resets the success streak but never counts toward a step-down, and leaves a failure streak running. Ceiling `1.0`, floor `0.20` (Stage 2's start, `VAR_THRESHOLD_FLOOR_FRACTION`). | Persists `threshold_fraction` + both streak counters every session, so a streak survives overnight. **Side-biased day → half the threshold next day** (since 2026-11-03): the first Stage 2 session of a day starts at half the saved threshold, with both streaks reset, if ≥ 20% (`VAR_BIAS_DAY_WITHHELD_TRIGGER`) of the previous Stage 2 day's turns were withheld (`session_csv_parser.previous_day_withheld()`). Runs once per day (`bias_day_checked`), saved immediately. VALs: `PREV_DAY_DATE`, `PREV_DAY_WITHHELD_FRACTION`, `THRESHOLD_BIAS_HALVED`, `THRESHOLD_START_BEFORE_HALVING_DEG`. **Day-start minimum 12.5°** (since 2026-11-05, `VAR_DAY_START_MIN_THRESHOLD_DEG`): the same once-a-day check raises a saved threshold below 12.5° up to it (streaks reset, `THRESHOLD_DAY_START_RAISED`), biased day or not, and halving never goes below it. Not applied on a subject's first Stage 2 day (no previous Stage 2 day: Stage 1's 7° hand-off stands). **Both only on a subject's first 3 Stage 2 days** (since 2026-11-06, `VAR_DAY_START_RULE_DAYS`, `session_csv_parser.protocol_days_before()`, VAL `STAGE2_DAY_NUMBER`); **From day 4 on, a day steps up** (since 2026-11-09): never halved, the first session starts at max(saved threshold, the previous day's peak `THRESHOLD_DEG`) + `DAY_START_STEP_FRACTION` (0.10 = 3.5°), ceiling 1.0, streaks reset (`later_day_start_fraction()`, `day_max_threshold_deg()`; VALs `PREV_DAY_MAX_THRESHOLD_DEG`, `THRESHOLD_DAY_START_STEPPED`). The within-session floor stays `0.20` (7°). |
| 3 / 4 | Frozen — read once at session start, never updated. Stage 3's reserve asymmetric threshold (see "Stage 3" below) scales it per side for a few sessions. | Passthrough only. If a subject somehow skipped Stage 2, Stages 3/4/5 all fall back to `staircase.THRESHOLD_FRACTION_DEFAULT` = `1.0` (one fallback since 2026-11-11; Stage 5 used to use 0.45). |
| 5 | Resumes the *same* `ThresholdStaircase` mechanism as Stage 2, continuing from whatever Stage 2/4 left persisted — but only on sessions where Tag B's selector (below) picks `'response'`; otherwise frozen for that whole session. | Target for advancement is `0.90` (`STAGE5_RESPONSE_THRESHOLD_FINAL_FRACTION`), **not** the staircase's own `1.0` ceiling — the staircase can still step past 0.90, advancement just stops requiring it to. |

## 2. ITI (inter-trial interval)

The correct/no-response-path wait before the next trial. (Error and abort trials get their own
fixed windows — see "Punishment and error/abort ITIs" below.)

| Stage | Within-session | Across-session |
|---|---|---|
| 1 | Fixed `0.5s`. | Never changes. |
| 2 | Fixed for the whole session. | Grows once per qualifying session (≥100 rewarded trials; same-day and Stopped sessions count too): `+0.1s`, ceiling `1.5s`. |
| 3 / 4 / 5 | **Fixed `VAR_ITI_S = 1.5s`** (Stage 2's final value), error ITI fixed `4.0s` (since 2026-11-11, spec A2). The Stage 3/4 staircase (1.5 → 3.0s on correct streaks) is gone: it cut the reward rate for good performance and shrank the error/correct contrast from 2.7× toward 1.33× while direction was being learned. | Read only: a saved `iti_s` above 1.5s (from that ramp) is registered as `ITI_RESET_FROM_S`. Stages 3-5 never write `iti_s` or its streak counters (Stage 1/2 own them). |

The correct-trial ITI starts at the first lick: `Consumption` ends on the first `Port1In` or after
3.0s. Harmless for training, but ITI-locked analyses must use the logged state times.

## 3. Cue / stimulus duration

The auditory click-train duration.

| Stage | Within-session | Across-session |
|---|---|---|
| 1 / 2 | No auditory stimulus. | — |
| 3 / 4 / 5 | **Fixed `0.35s`** (`staircase.CUE_DURATION_S`, since 2026-11-11, spec A6/B3): Ott et al. 2024's stimulus duration, ~7 clicks at 20Hz instead of 5. The Stage 3/4 cue staircase (0.25 → 2.0s) is gone: it added a second change while direction was being learned, and with one-sided clicks only added same-side clicks. Every trial's train has at least one click on its side inside the window (`click_train_v2.draw_trial_clicks_with_cue()`, `STIM_RESAMPLES` when it redrew). Stage 3's train keeps playing after the window (the click tail, see "Stage 3" below). | Read only: a saved `stim_duration_s` other than 0.35s is registered as `STIM_DURATION_RESET_FROM_S`; never written. Cue extension is its own stage (6, not built). |

## 4. Cue-to-go delay ("Tag A", `DelayPeriod`)

The gap between the cue ending and the response window opening.

| Stage | Within-session | Across-session |
|---|---|---|
| 1 / 2 | N/A — no cue, no delay period. | — |
| 3 / 4 / 5 | Fresh uniform random draw every trial, `[0.1s, 0.2s]`. **Not staircased at all.** | No persisted state — nothing to carry. |

Comment in code: "staircased 0-750ms only from Stage 7 onward, not built here." There is no
delay-staircase class anywhere in this codebase yet.

## 5. Hold-to-initiate (quiescence)

How long the wheel must sit still before a trial can start. `current_s` (q) is the **scale of
a truncated exponential draw**, not a fixed duration (since 2026-11-11, spec A1): the per-trial
hold is Exponential(scale `0.636·q`) truncated to `[0.727·q, 1.273·q]`
(`staircase.draw_quiescence_hold()`, inverse-CDF), mean ≈ `0.96·q`. At q = 0.55 that is IBL's own
hold (iblrig: `0.2 + truncated_exponential(0.35, 0.2, 0.5)`, i.e. 0.4–0.7s, mean 0.53s); at
q = 0.1 it is ~0.073–0.127s. The old `clip(Exponential(q), 0.1, 0.25)` put ~70% of trials exactly
on a bound. VALs: `QUIESCENCE_DUR_S` (the drawn hold), `QUIESCENCE_SCALE_S` (q).

| Stage | Within-session | Across-session |
|---|---|---|
| 1 | Fixed `0.1s` (100ms). | Never changes. |
| 2 | Fixed `0.1s` **until** a one-way latch unlocks it: the first time, within any Stage 2 session, the movement-threshold staircase (§1) steps *up* at least once (checked every trial, not just start-vs-end of session — a session that steps up and later back down still counts). Once unlocked, that same session still runs on the fixed 0.1s; staircasing begins the **next** session. | Persists `quiescence_unlocked` (one-way, never reset), and once unlocked, persists `quiescence_s` + both streak counters. |
| 2 (post-unlock) / 3 / 4 | Per-trial `QuiescenceStaircase`: `+0.025s` after 20 consecutive clean holds (zero resets), `-0.025s` after 5 consecutive holds with at least one reset. Floor `0.1s`, **ceiling `0.25s` in Stages 2-4** (`STAGE234_QUIESCENCE_CEILING_S`, spec B10: one new demand per stage, so q can't climb while the click → side rule is learned). | Continuous across Stage 2 → 3 → 4 once unlocked — one ramp, not three. A subject that somehow skips Stage 2 entirely falls back to the fixed `0.1s` in Stage 3/4 rather than crashing. |
| 5 | Continues the *same* persisted staircase up to the IBL ceiling **`0.55s`** (`QUIESCENCE_CEILING_S`). Only advances on sessions where Tag B's selector picks `'quiescence'`; frozen otherwise. 0.25 → 0.55s is 12 steps, at least 240 clean initiations. | At final values the enforced stillness is ≈ 0.55 (quiescence) + 2.75 (cue + delay, Stages 6-7) + 0.15 (J1) ≈ 3.45s. |

**The dot-onset jitter J1 (0.1-0.2s) is added on top of this, and is part of the hold
requirement, not a separate delay after it** (since 2026-10-22). The dot may only appear after
quiescence, and the wheel is zeroed the moment it does:
- Stage 1/2: the actual per-trial hold is `quiescence + J1` (`QUIESCENCE_HOLD_S`). The dot appears
  the instant the trial's first state starts.
- Stage 3/4/5: J1 is a native `DotOnsetDelay` state at the end of the cue machine, guarded by the
  same wheel-abort thresholds as the cue/delay.

The staircased value above, and the `n_breaks == 0` clean-hold criterion, are unchanged: in Stage
1/2 a break during the J1 part counts as a break like any other. J1 is the jitter between the end of
the delay and dot onset; the dot is the go cue (the LED is a punishment signal only).

**J2, threshold crossing → dot offset, is `0.3–0.6s` in every stage** (since 2026-11-11, spec A5;
was 0.4–0.9s), one window for both outcomes, from one constant
(`staircase.DOT_DISAPPEAR_MIN_S/MAX_S`). The 0.3s floor keeps the offset transient out of the
first few hundred ms after the crossing. Logged: `DOT_DISAPPEAR_JITTER_S` (drawn) and
`DOT_CLEAR_TIME` (painted); the audit checks the painted offset lands within 50ms of the draw.

**Tag B (Stage 5 only)**: at most one of {movement/response threshold, quiescence} may advance
in any given Stage 5 session — `select_stage5_staircase_to_tighten()` picks whichever is
furthest (as a normalized 0-1 fraction) from its own final value, tie-breaking by alternating
away from the previous session's pick (persisted as `stage5_last_staircase_advanced`). The
*inactive* one is fully frozen that session — its counters don't move either.

## Side-bias withholding (Stage 2 only)

`DirectionRatioTracker` (`direction_tracker.py`): the right-turn fraction over the last 40
movement directions (rewarded or withheld turns; no-movement trials don't count), recency-weighted
(half-Gaussian, `DIRECTION_RATIO_SIGMA = 8`, so ~10 effective trials, plus 2 neutral
pseudo-trials). Outside 30-70%, reward is withheld on the over-used side. From a balanced history
a same-side run is withheld after ~6 turns. Stage 1 and Stages 3-5 don't use it.

| Within-session | Across-session |
|---|---|
| Recomputed every trial. No withholding until 10 directions are known (`DIRECTION_RATIO_MIN_TRIALS`), counting the carried-over ones. `DIRECTION_RATIO` logs the weighted value, `DIRECTION_RATIO_FLAT` the unweighted one. | The last 40 directions persist as `recent_directions`, so withholding is live from trial 1 for an animal with Stage 2 history. `DIRECTION_HISTORY_SEEDED` logs how many were carried over. **Sigma grows +2 per day from Stage 2 day 4, ceiling 20** (since 2026-11-09, state `direction_sigma`, VAL `DIRECTION_SIGMA`, `grow_direction_sigma()`). Withholding then starts later in a run (6 same-side turns at sigma 8, 12 at sigma 20), and recovering from a long run takes more opposite turns (after 30: 3 at sigma 8, 7 at sigma 20). |

Before 2026-11-02 the window started empty every session and withholding waited for 40 trials of
that session, so UM001 (`20261102-160353`) got 22/22 left turns rewarded. The flat (unweighted)
ratio then let UM001 (`20261102-161142`) oscillate between long runs (R×31, L×29, …), each
rewarded for ~16 trials before the ratio left the band.

## Side debiasing (Stages 3-5)

Hu et al. (2009), `debiasing.HuDebiaser` (since 2026-11-11, spec B2; replaced "after an error,
re-present the same side with p = 0.80", which paid a lose-shift animal 62% without listening):
- e_R, e_L: half-Gaussian (sigma 20) error rates over the last 40 right / left trials.
- p_R = clip(√e_R) / (clip(√e_R) + clip(√e_L)), √ clipped to 0.15–0.85.
- f_R: the empirical right fraction (sigma 60, last 60 trials). Draw right with 0.5·p_R if
  f_R > p_R, else 0.5·(1 + p_R); 50/50 on an exact tie.
- Simulated: an always-right agent earns ~16%, win-stay/lose-shift ~49%.

| Within-session | Across-session |
|---|---|
| Recomputed before every fresh draw; records every responded trial except Stage 3's correction trials. The abort re-offer (same side again, `TRIAL_TYPE='repeat'`) is separate and stays. VALs `P_RIGHT_TARGET` (p_R), `P_RIGHT_DRAW`, `E_RIGHT`, `E_LEFT`, `F_RIGHT_EMPIRICAL`. | The last 200 responded trials persist as `hu_debias_history`, shared by Stages 3-5. Stage 2's `DirectionRatioTracker` is a separate mechanism. |

## Stage 3: fading scaffolds (since 2026-11-11, spec A7)

Random sides from the Hu debiaser from trial 1, no side blocks: within a block the last rewarded
side predicts the next as well as the clicks (win-stay/lose-shift scored 93–98% without
listening). Two aids fade one step per **qualifying session**: ≥ 100 responded main trials,
abort rate < 30%, click-weight criterion met (`stage3_scaffold.py`).

| Step | `P_CORR` (correction trials) | Click tail until |
|---|---|---|
| 0 | 0.8 | the crossing / timeout / abort |
| 1 | 0.6 | same |
| 2 | 0.4 | same |
| 3 | 0.2 | same |
| 4 | 0 | same |
| 5 | 0 | dot onset (silent response, as in Stage 4) |

| Within-session | Across-session |
|---|---|
| After an error the next trial repeats the side with p = `P_CORR`, fresh clicks (`TRIAL_TYPE='correction'`); correction trials are left out of accuracy, the regression, the sub-chance stop and the debiaser. The click train (generated for cue + delay + J1 + response window, ≈5.75s) keeps playing until the state that ends the response sends the HiFi STOP (`CueComplete` no longer does); `CLICK_STOP_S`, `N_CLICKS_PLAYED`, `N_CLICKS_CUE_L/R` (0.35s window). VALs `FADE_STEP`, `P_CORR`, `CLICK_TAIL_MODE`, `ASYM_STEP`. | `stage3_progress`: +1 step per qualifying session; 2 consecutive sessions with click z < 1 step back one; advance to Stage 4 after 2 consecutive qualifying sessions at step 5 (`STAGE3_QUALIFYING`, `FADE_STEP_NEXT`, `STAGE3_ADVANCE_READY`). **Reserve, asymmetric threshold**: never met the criterion by the end of session 4 at step 0 → the correct turn needs 40% of the threshold and the wrong turn 130%, fading to symmetric over 4 qualifying sessions (0.40/0.55/0.70/0.85, 1.30/1.225/1.15/1.075) before step 1; once only. Still never met by session 8: `STAGE3_FLAG_CHECK_SPEAKERS` (check speaker lateralisation and level at head position). |

## Click-weight criterion (Stages 3 and 4)

`choice_regression.py`: logit P(R) = b0 + b_click·S + b_prev·C₋₁ + b_win·C₋₁·Rw₋₁, fitted at
session end on main responded trials (history from every responded trial), ridge-penalised so a
never-wrong animal gets a finite weight. Met if b_click > 0, z ≥ 3 and b_click > max(|b_prev|,
|b_win|). VALs `CLICK_WEIGHT`, `CLICK_WEIGHT_Z`, `PREV_CHOICE_WEIGHT`, `WIN_STAY_WEIGHT`,
`CLICK_WEIGHT_N`, `CLICK_CRITERION_MET`. Stage 4 now advances on > 70% correct, abort rate < 20%
**and** the criterion, 3 sessions in a row.

## Warm-up (Stages 3-5)

At least 20 trials, then until the last 10 are ≥ 85% correct with ≤ 10 points between left and
right accuracy (over the last 20), at most 60 (`staircase.WarmupGate`, since 2026-11-11, spec B7,
after Pinto et al. 2018). `WARMUP_EXIT_REASON` ('criterion'/'cap') and `WARMUP_TRIALS`.

## Click attenuation (Stages 3/4)

−6dB for the first `VAR_CLICK_ATTENUATION_SESSIONS` (2) full sessions. Since 2026-11-11 a
session only counts down when it had ≥ 100 responded main trials, so a crashed or stopped-early
session doesn't use one up (spec B9).

## Session caps (Stage 1/2)

| Cap | Scope | Value |
|---|---|---|
| Trials attempted | per session | `VAR_TRIAL_CAP`: Stage 1 250, Stage 2 300 (since 2026-11-05) |
| Time | per session | 30 min |

Whichever comes first ends the session. The 1.5 mL daily water cap was removed on 2026-11-06
(per explicit instruction). Consumed water is still logged (`SESSION_WATER_CONSUMED_UL`,
`DAY_WATER_CONSUMED_UL`, `WATER_CONSUMED_EARLIER_TODAY_UL`); older sessions may show
`SESSION_END_REASON=water_cap`.

## Punishment and error/abort ITIs (not staircased, any stage)

The 350ms LED + full-white-screen flash on errors and aborts is **deliberate**: bright light is
aversive and doesn't interfere with the auditory cues; there is no auditory error cue. On choice
trials it follows errors only, so it is confounded with the error outcome and must be its own
event regressor in any neural model.

- **Wrong-direction wheel turn** (Stages 3/4/5 only — Stages 1/2 have no correct-side concept at
  all): `ErrorConsumption` → `ErrorPunish` (350ms flash) → `IncorrectITI` (fixed
  `VAR_INCORRECT_ITI_S = 4.0s`). Since 2026-11-11 (spec B8) `ErrorConsumption` lasts a per-trial
  U(0.7, 1.0)s (`ERROR_FLASH_DELAY_S`) and a lick no longer ends it: the flash used to come at the
  first lick or 0.5s, locked to lick onset and overlapping J2. 0.7s is 100ms after the latest dot
  offset.
- **Mid-cue wheel-movement abort** (Stages 3/4/5 only): `WheelAbort` → `AbortPunish` (350ms
  flash) → `AbortITI` = `VAR_INCORRECT_ITI_S` (4.0s, since 2026-11-11, spec A4; was 1–2s, which
  made aborting the cheaper exit from a hard trial). `ABORT_ITI_S` per abort. The same side is
  re-offered with fresh clicks (`TRIAL_TYPE='repeat'`).
- A plain timeout (`NoResponse`, no choice made at all) gets neither the punishment flash nor
  either of these — it proceeds straight to the normal ITI (§2).
- None of these durations are staircased or persisted — they're fixed per-stage constants.

## Known quirks (not fixed here)

These are real, confirmed behaviors of the current code — documented so they're not mistaken for
bugs during a future change, and flagged in case they're worth revisiting on their own:

Resolved on 2026-11-11 and removed from this list: Stage 5 discarding the Stage 3/4 ITI and cue
ramps (both ramps are gone), the inconsistent `threshold_fraction` fallback, the attenuation
countdown on every session run, and Stage 5's quiescence gate already met on its first session
(Stages 2-4 now stop at 0.25s).

- **Qualifying-session criteria are not uniform**: Stage 1/2 count every completed trial; Stage
  3/4/5 count only `main` trials with a genuine L/R response. The number of consecutive qualifying
  sessions required to advance also differs: 2 (Stage 1), human-judgment-call (Stage 2 — no
  automatic "advance-ready" check exists), 2 at the last fade step (Stage 3), 3 (Stage 4), 2
  (Stage 5).
- **Stage 3's click tail is a ≈5.75s stereo buffer at 192kHz (~4.4MB per trial)**, loaded between
  machines. Not yet measured on the rig: check it fits `hifi.max_samples_per_waveform` and time
  the load before the first animal runs Stage 3.
