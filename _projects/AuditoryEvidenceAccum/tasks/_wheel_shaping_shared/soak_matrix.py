"""Runs soak.py over a matrix of fake-mouse profiles and prints one summary line per run.

Usage:
  python soak_matrix.py matrix [n_trials] [stage ...]       every profile x every stage (default 3 4)
  python soak_matrix.py chain <stage> <profile> <n_sessions> [n_trials]
                                                            sessions back to back on one state file

Each line: audit result, trials with a choice, accuracy, fraction of R trials, warm-up exit,
regression z (CLICK_WEIGHT_Z), and the Stage 3 fade/advance values. Full logs are kept next to the
summary (path printed at the end). No hardware, nothing written into the repo.
"""
import csv, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.dirname(HERE)
STAGES = {'3': 'stage3_tone_side_association', '4': 'stage4_clicks_direction',
          '5': 'stage5_resume_staircases'}
PROFILES = {
    'random':    {},
    'listener':  {'SOAK_P_CORRECT': '0.9'},
    'biased':    {'SOAK_P_LEFT': '0.9'},
    'repeater':  {'SOAK_P_REPEAT': '0.9'},
    'nomove':    {'SOAK_P_NOMOVE': '1'},
    'impulsive': {'SOAK_P_EARLY': '1', 'SOAK_P_CUE_ABORT': '0.9'},
}
PARALLEL = 3   # more than this and the audit's render/timing budgets start to measure CPU contention


def script(stage):
    name = STAGES[stage]
    return os.path.join(TASKS, name, name + '.py')


def vals(csv_path):
    """{key: [values in order]} from a soak CSV's VAL rows."""
    out = {}
    with open(csv_path, encoding='utf-8') as f:
        for r in csv.reader(f, delimiter=';'):
            if r and r[0] == 'VAL' and len(r) > 5:
                out.setdefault(r[4], []).append(r[5])
    return out


def last(v, k):
    return v.get(k, ['-'])[-1]


def summarize(label, log_path):
    with open(log_path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    folder = [l.split(' ', 1)[1].strip() for l in text.splitlines() if l.startswith('SOAK_FOLDER')]
    audit = 'ALL PASS' if '-> ALL PASS' in text else ('FAIL' if '[FAIL]' in text else 'no audit')
    if 'Traceback' in text:
        audit += ' +traceback'
    if not folder:
        return '%-22s %s (no session folder; see %s)' % (label, audit, log_path)
    v = vals(os.path.join(folder[0], 'SOAK.csv'))
    correct = [c for c in v.get('CORRECT', []) if c in ('True', 'False')]
    sides = v.get('TRIAL_SIDE', [])
    acc = '%.0f%%' % (100.0 * correct.count('True') / len(correct)) if correct else '-'
    right = '%.0f%%' % (100.0 * sides.count('R') / len(sides)) if sides else '-'
    parts = ['%-22s %-9s' % (label, audit), 'choices=%-4d' % len(correct), 'acc=%-4s' % acc,
             'R=%-4s' % right, 'warmup=%s/%s' % (last(v, 'WARMUP_EXIT_REASON'), last(v, 'WARMUP_TRIALS')),
             'z=%s' % last(v, 'CLICK_WEIGHT_Z'), 'crit=%s' % last(v, 'CLICK_CRITERION_MET'),
             'end=%s' % last(v, 'SESSION_END_REASON')]
    if 'FADE_STEP' in v:
        parts += ['fade=%s->%s' % (last(v, 'FADE_STEP'), last(v, 'FADE_STEP_NEXT')),
                  'p_corr=%s' % last(v, 'P_CORR'), 'qual=%s' % last(v, 'STAGE3_QUALIFYING'),
                  'advance=%s' % last(v, 'STAGE3_ADVANCE_READY')]
    return '  '.join(parts)


def run(stage, profile, n_trials, seed, out_dir, state_file=None, tag=''):
    label = 'S%s %s%s s%d' % (stage, profile, tag, seed)
    log_path = os.path.join(out_dir, label.replace(' ', '_') + '.log')
    env = dict(os.environ, **PROFILES[profile])
    if state_file:
        env['SOAK_STATE_FILE'] = state_file
    with open(log_path, 'w', encoding='utf-8') as log:
        subprocess.call([sys.executable, os.path.join(HERE, 'soak.py'), script(stage), str(n_trials),
                         str(seed)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=out_dir)
    line = summarize(label, log_path)
    print(line, flush=True)
    return line


def main(argv):
    out_dir = tempfile.mkdtemp(prefix='soak_matrix_')
    t0 = time.time()
    if argv and argv[0] == 'chain':
        stage, profile, n_sessions = argv[1], argv[2], int(argv[3])
        # soak ends a session at n * 3s (its compute_remaining_quota stub), about 0.4 choices per
        # n. Stage 3 needs >= 100 responded MAIN trials to qualify (warm-up and correction trials
        # don't count), so a chain needs n ~ 450; 320 qualified on day 1 only, by a few trials.
        n_trials = int(argv[4]) if len(argv) > 4 else 450
        state_file = os.path.join(out_dir, 'state.json')
        for i in range(n_sessions):
            run(stage, profile, n_trials, 100 + i, out_dir, state_file, tag=' day%d' % (i + 1))
        print('final state:', json.dumps(json.load(open(state_file)), sort_keys=True)
              if os.path.exists(state_file) else 'none written')
    else:
        n_trials = int(argv[1]) if len(argv) > 1 else 150
        stages = argv[2:] or ['3', '4']
        jobs = [(s, p, seed) for s in stages for p in PROFILES for seed in (1, 2)]
        with ThreadPoolExecutor(PARALLEL) as ex:
            list(ex.map(lambda j: run(j[0], j[1], n_trials, j[2], out_dir), jobs))
    print('logs in', out_dir, '(%.0f min)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main(sys.argv[1:])
