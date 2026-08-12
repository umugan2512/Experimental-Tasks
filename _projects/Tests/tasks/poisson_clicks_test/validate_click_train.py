# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Offline validation of click_train.py -- no Bpod/hardware involved. Simulates many trials per
difficulty level and checks:
  1. realized click counts/rates and count-difference (delta) against the nominal grid
  2. how "Poisson" the floored inter-click-interval process actually is
  3. that the difficulty-level and side schedulers draw at their intended frequencies

Run with the pybpod-environment interpreter:
    /c/Users/2P-Behav/.conda/envs/pybpod-environment/python.exe validate_click_train.py
"""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import click_train as ct

RNG = np.random.RandomState(12345)
N_SIM_PER_LEVEL = 3000       # trials simulated per difficulty level, for rate/ISI stats
N_SCHEDULE_DRAWS = 20000     # draws used to check draw_difficulty()/draw_side() frequencies
FLOOR_TOL_S = 1e-9           # tolerance for "isi == floor" (clipped) detection


def simulate_rate_stats(r_high, r_low, n_trials):
    """ Bypasses side/L-R mapping -- generates the high-rate and low-rate streams directly, so
    rate/Poisson-ness stats aren't entangled with the (separately validated) L/R side mapping.
    r_high/r_low are NOMINAL target rates; generation uses the calibrated attempt rate so the
    realized rate lands on the nominal target (see click_train.calibrated_rate()). """
    n_high, n_low = [], []
    isi_high, isi_low = [], []
    attempt_high = ct.calibrated_rate(r_high)
    attempt_low = ct.calibrated_rate(r_low)
    for _ in range(n_trials):
        t_high = ct.generate_floored_poisson_train(attempt_high, ct.VAR_STIM_DURATION_S, ct.VAR_ISI_FLOOR_S, RNG)
        t_low = ct.generate_floored_poisson_train(attempt_low, ct.VAR_STIM_DURATION_S, ct.VAR_ISI_FLOOR_S, RNG)
        n_high.append(len(t_high))
        n_low.append(len(t_low))
        if len(t_high):
            isi_high.extend(np.diff(np.concatenate(([0.0], t_high))))
        if len(t_low):
            isi_low.extend(np.diff(np.concatenate(([0.0], t_low))))
    return {
        'n_high': np.array(n_high), 'n_low': np.array(n_low),
        'isi_high': np.array(isi_high), 'isi_low': np.array(isi_low),
        'attempt_high': attempt_high, 'attempt_low': attempt_low,
    }


def isi_fidelity(isi, rate_hz):
    """ Fraction of drawn ISIs that hit the floor (clipped), and a KS test of the non-clipped
    ISIs (minus the floor) against Exponential(rate_hz) -- valid by the memoryless property:
    an exponential draw conditioned on exceeding the floor is itself floor + Exponential(rate). """
    if len(isi) == 0:
        return {'frac_clipped': float('nan'), 'ks_stat': float('nan'), 'ks_p': float('nan'),
                'realized_rate': float('nan')}
    clipped = isi <= (ct.VAR_ISI_FLOOR_S + FLOOR_TOL_S)
    frac_clipped = clipped.mean()
    tail = isi[~clipped] - ct.VAR_ISI_FLOOR_S
    if len(tail) > 20:
        ks_stat, ks_p = stats.kstest(tail, 'expon', args=(0, 1.0 / rate_hz))
    else:
        ks_stat, ks_p = float('nan'), float('nan')
    realized_rate = 1.0 / isi.mean()
    return {'frac_clipped': frac_clipped, 'ks_stat': ks_stat, 'ks_p': ks_p,
            'realized_rate': realized_rate}


# --- 1+2: per-level rate/count/ISI stats ---------------------------------------------------------

per_level = {}
for level in ct.DIFFICULTY_ORDER:
    grid = ct.DIFFICULTY_GRID[level]
    r_high, r_low = grid['r_high'], grid['r_low']
    sim = simulate_rate_stats(r_high, r_low, N_SIM_PER_LEVEL)
    delta = sim['n_high'] - sim['n_low']
    per_level[level] = {
        'r_high': r_high, 'r_low': r_low, 'gamma': grid['gamma'],
        'attempt_high': sim['attempt_high'], 'attempt_low': sim['attempt_low'],
        'nominal_delta': (r_high - r_low) * ct.VAR_STIM_DURATION_S,
        'delta_mean': delta.mean(), 'delta_sd': delta.std(),
        'n_high_mean': sim['n_high'].mean(), 'n_high_var': sim['n_high'].var(),
        'n_low_mean': sim['n_low'].mean(), 'n_low_var': sim['n_low'].var(),
        'fidelity_high': isi_fidelity(sim['isi_high'], sim['attempt_high']),
        'fidelity_low': isi_fidelity(sim['isi_low'], sim['attempt_low']) if r_low > 0 else None,
        'delta_raw': delta,
        'isi_high': sim['isi_high'],
    }

print("=" * 130)
print("Click-train stats: N={0} sim trials/level, lambda={1}Hz, T={2}s, ISI floor={3}ms, click={4}ms"
      .format(N_SIM_PER_LEVEL, ct.VAR_TOTAL_RATE_HZ, ct.VAR_STIM_DURATION_S,
              ct.VAR_ISI_FLOOR_S * 1000, ct.VAR_CLICK_DURATION_S * 1000))
print("(nom_rH/nom_rL = target rate; attempt_H/attempt_L = calibrated rate fed to the sampler;")
print(" real_rH/real_rL = realized rate after flooring -- should now track nom_rH/nom_rL closely)")
print("=" * 130)
print("{:6s} {:>7s} {:>9s} {:>9s} {:>9s} {:>7s} {:>9s} {:>9s} {:>9s} {:>7s} {:>9s} {:>10s} {:>10s}".format(
    "level", "gamma", "nom_rH", "attmpt_H", "real_rH", "fano_H", "nom_rL", "attmpt_L", "real_rL", "fano_L",
    "nom_dlt", "real_dlt", "dlt_sd"))
for level in ct.DIFFICULTY_ORDER:
    p = per_level[level]
    real_rH = p['fidelity_high']['realized_rate']
    fano_H = p['n_high_var'] / p['n_high_mean'] if p['n_high_mean'] > 0 else float('nan')
    if p['r_low'] > 0:
        real_rL = p['fidelity_low']['realized_rate']
        fano_L = p['n_low_var'] / p['n_low_mean'] if p['n_low_mean'] > 0 else float('nan')
    else:
        real_rL, fano_L = 0.0, float('nan')
    print("{:6s} {:7.3f} {:9.3f} {:9.3f} {:9.3f} {:7.3f} {:9.3f} {:9.3f} {:9.3f} {:7.3f} {:9.2f} {:10.2f} {:10.2f}".format(
        level, p['gamma'] if p['gamma'] != float('inf') else float('nan'),
        p['r_high'], p['attempt_high'], real_rH, fano_H,
        p['r_low'], p['attempt_low'], real_rL, fano_L,
        p['nominal_delta'], p['delta_mean'], p['delta_sd']))

print()
print("ISI floor fidelity (fraction of intervals clipped to the floor, and KS-test of the")
print("non-clipped tail against a pure Exponential(rate) -- high p-value = statistically")
print("indistinguishable from Poisson once you account for the floor):")
print("{:6s} {:>14s} {:>10s} {:>10s} {:>14s} {:>10s} {:>10s}".format(
    "level", "frac_clip_H", "ks_stat_H", "ks_p_H", "frac_clip_L", "ks_stat_L", "ks_p_L"))
for level in ct.DIFFICULTY_ORDER:
    p = per_level[level]
    fh = p['fidelity_high']
    fl = p['fidelity_low']
    if fl is not None:
        print("{:6s} {:14.3f} {:10.3f} {:10.3f} {:14.3f} {:10.3f} {:10.3f}".format(
            level, fh['frac_clipped'], fh['ks_stat'], fh['ks_p'],
            fl['frac_clipped'], fl['ks_stat'], fl['ks_p']))
    else:
        print("{:6s} {:14.3f} {:10.3f} {:10.3f} {:>14s} {:>10s} {:>10s}".format(
            level, fh['frac_clipped'], fh['ks_stat'], fh['ks_p'], "n/a(r=0)", "-", "-"))

# --- 3: difficulty/side scheduler frequency checks ------------------------------------------------

difficulty_draws = [ct.draw_difficulty(RNG) for _ in range(N_SCHEDULE_DRAWS)]
difficulty_counts = {lvl: difficulty_draws.count(lvl) / N_SCHEDULE_DRAWS for lvl in ct.DIFFICULTY_ORDER}

side_draws = [ct.draw_side(RNG) for _ in range(N_SCHEDULE_DRAWS)]
side_counts = {s: side_draws.count(s) / N_SCHEDULE_DRAWS for s in ('L', 'R')}

print()
print("Difficulty draw frequency (N={0} draws) vs. intended weights:".format(N_SCHEDULE_DRAWS))
print("{:6s} {:>10s} {:>10s} {:>10s}".format("level", "nominal", "realized", "diff"))
for level in ct.DIFFICULTY_ORDER:
    nom = ct.DIFFICULTY_WEIGHTS[level]
    real = difficulty_counts[level]
    print("{:6s} {:10.4f} {:10.4f} {:10.4f}".format(level, nom, real, real - nom))

print()
print("Side draw frequency (N={0} draws) vs. 50/50:".format(N_SCHEDULE_DRAWS))
for side in ('L', 'R'):
    print("  {0}: nominal=0.5000  realized={1:.4f}  diff={2:.4f}".format(
        side, side_counts[side], side_counts[side] - 0.5))

# --- 4: end-to-end L/R mapping sanity check via generate_trial_clicks -----------------------------

print()
print("End-to-end L/R mapping check (generate_trial_clicks with random side draws, N=1000/level):")
print("{:6s} {:>10s} {:>12s} {:>12s}".format("level", "side", "mean_n_high_side", "mean_n_low_side"))
for level in ct.DIFFICULTY_ORDER:
    n_when_R_is_right, n_when_L_is_left = [], []
    n_low_when_R, n_low_when_L = [], []
    for _ in range(1000):
        side = ct.draw_side(RNG)
        trial = ct.generate_trial_clicks(level, side, RNG)
        if side == 'R':
            n_when_R_is_right.append(trial['n_right'])
            n_low_when_R.append(trial['n_left'])
        else:
            n_when_L_is_left.append(trial['n_left'])
            n_low_when_L.append(trial['n_right'])
    print("{:6s} {:>10s} {:12.2f} {:12.2f}".format(
        level, "R", np.mean(n_when_R_is_right) if n_when_R_is_right else float('nan'),
        np.mean(n_low_when_R) if n_low_when_R else float('nan')))
    print("{:6s} {:>10s} {:12.2f} {:12.2f}".format(
        level, "L", np.mean(n_when_L_is_left) if n_when_L_is_left else float('nan'),
        np.mean(n_low_when_L) if n_low_when_L else float('nan')))

# --- plots -----------------------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) realized delta per level, box plot, nominal delta marked
ax = axes[0, 0]
box_data = [per_level[level]['delta_raw'] for level in ct.DIFFICULTY_ORDER]
ax.boxplot(box_data, labels=ct.DIFFICULTY_ORDER, showfliers=False)
nominal_deltas = [per_level[level]['nominal_delta'] for level in ct.DIFFICULTY_ORDER]
ax.plot(range(1, len(ct.DIFFICULTY_ORDER) + 1), nominal_deltas, 'r_', markersize=20, markeredgewidth=2,
        label='nominal E|delta|')
ax.set_xlabel('difficulty level')
ax.set_ylabel('realized n_high - n_low')
ax.set_title('Realized click-count difference vs. nominal')
ax.legend()

# (b) ISI histogram for the fastest stream (AOS high rate) vs theoretical exponential
ax = axes[0, 1]
isi_aos = per_level['AOS']['isi_high']
bins = np.linspace(0, isi_aos.max() if len(isi_aos) else 0.3, 60)
ax.hist(isi_aos, bins=bins, density=True, alpha=0.6,
        label='simulated ISIs (AOS, nominal={0:.1f}Hz, attempt={1:.2f}Hz)'.format(
            per_level['AOS']['r_high'], per_level['AOS']['attempt_high']))
x = np.linspace(ct.VAR_ISI_FLOOR_S, bins[-1], 200)
rate = per_level['AOS']['attempt_high']
ax.plot(x, rate * np.exp(-rate * (x - ct.VAR_ISI_FLOOR_S)), 'r-',
        label='theoretical Exp(attempt rate), shifted by floor')
ax.axvline(ct.VAR_ISI_FLOOR_S, color='k', linestyle='--', label='ISI floor ({0:.0f}ms)'.format(
    ct.VAR_ISI_FLOOR_S * 1000))
ax.set_xlabel('inter-click interval (s)')
ax.set_ylabel('density')
ax.set_title('ISI distribution vs. floored-exponential (fastest stream)')
ax.legend(fontsize=8)

# (c) difficulty draw frequency vs nominal
ax = axes[1, 0]
x = np.arange(len(ct.DIFFICULTY_ORDER))
width = 0.35
nom_w = [ct.DIFFICULTY_WEIGHTS[level] for level in ct.DIFFICULTY_ORDER]
real_w = [difficulty_counts[level] for level in ct.DIFFICULTY_ORDER]
ax.bar(x - width / 2, nom_w, width, label='nominal')
ax.bar(x + width / 2, real_w, width, label='realized')
ax.set_xticks(x)
ax.set_xticklabels(ct.DIFFICULTY_ORDER)
ax.set_ylabel('draw frequency')
ax.set_title('Difficulty-level draw frequency vs. intended weights')
ax.legend()

# (d) side draw frequency vs 50/50
ax = axes[1, 1]
sides = ['L', 'R']
nom_s = [0.5, 0.5]
real_s = [side_counts[s] for s in sides]
x = np.arange(2)
ax.bar(x - width / 2, nom_s, width, label='nominal')
ax.bar(x + width / 2, real_s, width, label='realized')
ax.set_xticks(x)
ax.set_xticklabels(sides)
ax.set_ylim(0, 1)
ax.set_ylabel('draw frequency')
ax.set_title('Side draw frequency vs. 50/50')
ax.legend()

fig.tight_layout()
out_path = 'click_train_validation.png'
fig.savefig(out_path, dpi=150)
print()
print("Saved plots to {0}".format(out_path))
