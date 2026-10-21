"""Commit and share this lab's DATA (training_log.xlsx + session data) on main, from any branch.

Data = `records/training_log.xlsx` plus the session folders under `AuditoryEvidenceAccum/experiments/`
(session CSV, `_struct.json`/`.mat`, video timestamps -- `.gitignore` keeps out videos, preview
JPGs, launch logs, session-local user_settings.py, and the box-specific setup `.json` files).

Two entry points:
- `commit_data_locally()` -- called after every weight/notes Save in run_session.py. Commits the
  data onto the LOCAL `main` branch only; never pushes.
- `sync()` -- run via `Sync Training Log.bat`: rebuild training_log.xlsx, commit the data, bring in
  every other computer's latest main, and push main.

**Nothing another computer wrote is ever overwritten.** Every computer commits to the same main,
under the rule "one mouse, one computer":
- build_training_log.py only rewrites the sheets (and session folders) of mice this computer owns
  (`build_training_log.owned_subjects()`: the mouse's most recent session ran here).
- Only session files in folders that ran on this computer are committed (`_ran_here()`); a local
  change to another computer's session file is left out with a warning. Local deletions are never
  committed (`git add --ignore-removal`).
- When another computer pushed in the meantime, the two mains are merged here
  (`reconcile_with_origin()`): session files are unioned, refusing (not overwriting) if the same file
  differs on both sides, and training_log.xlsx is merged sheet by sheet (`xlsx_sheet_merge.py`):
  their version of every sheet, except this computer's own mice's sheets.

Works the same whichever branch is checked out:
- On `main`: the data is committed straight onto main.
- On a computer-specific branch (e.g. `training05`: main plus only that machine's own patches,
  such as interpreter paths -- all real code lives on main): main is never checked out. The data
  commit is built on main via a temporary index, and a matching merge commit (main -> this
  branch) is written by hand, so the files already on disk become tracked without git's own
  merge refusing to overwrite them. Only data ever lands on main from such a branch.
"""
import functools
import glob
import os
import subprocess
import sys
import tempfile

import build_training_log
import xlsx_sheet_merge

RECORDS_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_NAME = 'training_log.xlsx'
# Relative to RECORDS_DIR (every git call runs from there).
DATA_PATHS = [XLSX_NAME, os.path.join('..', 'experiments')]
MAIN = 'main'
REMOTE_MAIN = 'origin/' + MAIN


class SyncError(Exception):
    pass


def git(*args, env=None, check=True):
    proc = subprocess.run(['git'] + list(args), cwd=RECORDS_DIR, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if check and proc.returncode != 0:
        raise SyncError('git {0} failed:\n{1}{2}'.format(' '.join(args), proc.stdout, proc.stderr))
    return proc


def git_out(*args, **kwargs):
    return git(*args, **kwargs).stdout.strip()


def git_bytes(*args, stdin=None):
    """Binary-safe git call (blob contents)."""
    proc = subprocess.run(['git'] + list(args), cwd=RECORDS_DIR, input=stdin,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise SyncError('git {0} failed:\n{1}'.format(' '.join(args), proc.stderr.decode(errors='replace')))
    return proc.stdout


def is_ancestor(a, b):
    return git('merge-base', '--is-ancestor', a, b, check=False).returncode == 0


def blob_at(rev, repo_path):
    """Blob hash of repo_path at rev, or None if it doesn't exist there."""
    proc = git('rev-parse', '--verify', '-q', '{0}:{1}'.format(rev, repo_path), check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


@functools.lru_cache(maxsize=None)
def repo_root():
    return git_out('rev-parse', '--show-toplevel')


def xlsx_repo_path():
    return git_out('rev-parse', '--show-prefix') + XLSX_NAME


def experiments_repo_prefix():
    records_prefix = git_out('rev-parse', '--show-prefix').rstrip('/')
    return records_prefix.rsplit('/', 1)[0] + '/experiments/'


def _ran_here(repo_path):
    """True if the session folder holding repo_path was run on this computer (see
    build_training_log._ran_here())."""
    return os.path.exists(os.path.join(repo_root(), os.path.dirname(repo_path), 'user_settings.py'))


def _session_subject(repo_path):
    """Subject name for the session folder holding repo_path (any file inside it), by reading
    that session's own CSV -- or None if no readable CSV is found there. Used to keep bench-test
    subjects' session data out of what gets committed to GitHub, per explicit instruction (see
    build_training_log.is_real_animal_subject())."""
    session_dir = os.path.join(repo_root(), os.path.dirname(repo_path))
    csvs = glob.glob(os.path.join(session_dir, '*.csv'))
    if not csvs:
        return None
    try:
        return build_training_log.session_subject(csvs[0])
    except Exception:
        return None


def build_log():
    print('Refreshing training_log.xlsx from local session data...', flush=True)
    rc = subprocess.call([sys.executable, 'build_training_log.py'], cwd=RECORDS_DIR)
    if rc != 0:
        raise SyncError('build_training_log.py failed -- see the message above. Nothing was committed.')


def commit_message():
    return 'Update training data from {0}'.format(os.environ.get('COMPUTERNAME', 'unknown box'))


def tree_with_data(base_ref):
    """The tree of `base_ref` plus the working-tree data files (added/modified only, never
    removed), built in a temporary index so HEAD, the real index and every other file are
    untouched. Changes to session files in folders that did NOT run on this computer, OR whose
    subject isn't a real training animal (build_training_log.is_real_animal_subject() -- per
    explicit instruction, bench-test session data is never committed to GitHub), are left out, so
    another computer's sessions are never overwritten from here and test data never leaves this
    machine. Returns the tree hash."""
    fd, tmp_index = tempfile.mkstemp(prefix='sync_training_log_index_')
    os.close(fd)
    os.remove(tmp_index)  # git wants to create the index file itself
    try:
        env = dict(os.environ, GIT_INDEX_FILE=tmp_index)
        git('read-tree', base_ref, env=env)
        git('add', '--ignore-removal', '--', *DATA_PATHS, env=env)
        exp_prefix = experiments_repo_prefix()
        changed = git_out('diff-index', '--cached', '--name-only', base_ref, env=env).splitlines()
        for path in changed:
            if not path.startswith(exp_prefix):
                continue
            if not _ran_here(path):
                reason = 'that session ran on another computer'
            elif not build_training_log.is_real_animal_subject(_session_subject(path)):
                reason = 'not a real training animal (bench-test subject)'
            else:
                continue
            print('WARNING: not committing {0} -- {1}.'.format(path, reason), flush=True)
            base_blob = blob_at(base_ref, path)
            if base_blob:
                git('update-index', '--cacheinfo', '100644,{0},{1}'.format(base_blob, path), env=env)
            else:
                git('update-index', '--force-remove', '--', path, env=env)
        return git_out('write-tree', env=env)
    finally:
        if os.path.exists(tmp_index):
            os.remove(tmp_index)


def commit_data_locally():
    """Commit the data onto the LOCAL `main` branch -- never pushes -- from whichever branch is
    checked out (see the module docstring). Returns a one-line status message; raises SyncError
    on failure."""
    branch = git_out('rev-parse', '--abbrev-ref', 'HEAD')
    if branch == 'HEAD':
        raise SyncError('Detached HEAD -- check out main or this computer\'s branch first.')

    if branch != MAIN and not is_ancestor(MAIN, 'HEAD'):
        # The hand-written merge below is only correct if this branch already has all of main.
        if git('merge', '--no-edit', MAIN, check=False).returncode != 0:
            git('merge', '--abort', check=False)
            raise SyncError('"{0}" is behind {1} and merging it in failed (usually because '
                            'training_log.xlsx was edited here while {1} has a newer one). Nothing '
                            'was committed; your edits are still in the files.'.format(branch, MAIN))

    old_main = git_out('rev-parse', MAIN)
    main_tree = tree_with_data(old_main)
    if main_tree == git_out('rev-parse', old_main + '^{tree}'):
        return 'No new data -- nothing to commit.'
    data_commit = git_out('commit-tree', main_tree, '-p', old_main, '-m', commit_message())

    if branch == MAIN:
        git('update-ref', 'refs/heads/' + MAIN, data_commit, old_main)
    else:
        old_head = git_out('rev-parse', 'HEAD')
        merge_commit = git_out('commit-tree', tree_with_data(old_head), '-p', old_head,
                               '-p', data_commit,
                               '-m', 'Merge branch \'{0}\' into {1}'.format(MAIN, branch))
        git('update-ref', 'refs/heads/' + MAIN, data_commit, old_main)
        git('update-ref', 'refs/heads/' + branch, merge_commit, old_head)
    # Bring the real index in line with the new HEAD for the data paths only (files on disk
    # already match it); anything else the user had staged is left alone.
    git('reset', '-q', '--', *DATA_PATHS, check=False)
    return 'Committed training data to {0} (not pushed).'.format(MAIN)


def _merge_mains(base):
    """Merge commit of origin/main and local main, where local main's own changes since `base`
    are data only. Returns the new commit hash. Never overwrites anything: see the module
    docstring."""
    xlsx_path = xlsx_repo_path()
    exp_prefix = experiments_repo_prefix()
    changed = git_out('diff', '--name-only', base, MAIN).splitlines()
    not_data = [p for p in changed if p != xlsx_path and not p.startswith(exp_prefix)]
    if not_data:
        raise SyncError('Local {0} has unpushed commits that are not training data ({1}), and '
                        'origin/{0} has new commits too. Nothing was updated or pushed -- merge '
                        'those by hand (git switch {0}; git pull), then run again.'
                        .format(MAIN, ', '.join(not_data[:5])))

    fd, tmp_index = tempfile.mkstemp(prefix='sync_training_log_index_')
    os.close(fd)
    os.remove(tmp_index)
    try:
        env = dict(os.environ, GIT_INDEX_FILE=tmp_index)
        git('read-tree', REMOTE_MAIN, env=env)
        for path in changed:
            ours = blob_at(MAIN, path)
            theirs = blob_at(REMOTE_MAIN, path)
            original = blob_at(base, path)
            if ours is None or ours == theirs or ours == original:
                continue   # nothing of ours to add (data is never deleted) / already identical
            if theirs is not None and theirs != original:
                if path != xlsx_path:
                    raise SyncError('{0} was changed both here and on another computer -- not '
                                    'overwriting either. Nothing was pushed; sort it out by hand.'
                                    .format(path))
                # Both computers changed the workbook: merge it sheet by sheet.
                our_sheets = {build_training_log._sheet_name_for(s)
                              for s in build_training_log.owned_subjects()}
                print('Merging training_log.xlsx with the other computers\' version (this '
                      'computer\'s sheets: {0})...'.format(', '.join(sorted(our_sheets)) or 'none'),
                      flush=True)
                merged = xlsx_sheet_merge.merge_workbooks(
                    git_bytes('cat-file', 'blob', theirs), git_bytes('cat-file', 'blob', ours),
                    our_sheets)
                ours = git_bytes('hash-object', '-w', '--stdin', stdin=merged).decode().strip()
            git('update-index', '--add', '--cacheinfo', '100644,{0},{1}'.format(ours, path), env=env)
        tree = git_out('write-tree', env=env)
    finally:
        if os.path.exists(tmp_index):
            os.remove(tmp_index)
    return git_out('commit-tree', tree, '-p', REMOTE_MAIN, '-p', MAIN, '-m',
                   'Merge {0} into training data from {1}'.format(
                       REMOTE_MAIN, os.environ.get('COMPUTERNAME', 'unknown box')))


def reconcile_with_origin(branch):
    """Bring local main (and the checked-out branch) up to date with origin/main without
    overwriting anything either side wrote. Call right after `git fetch`."""
    if is_ancestor(REMOTE_MAIN, MAIN):
        target = None                      # nothing new on origin
    elif is_ancestor(MAIN, REMOTE_MAIN):
        target = git_out('rev-parse', REMOTE_MAIN)
    else:
        target = _merge_mains(git_out('merge-base', MAIN, REMOTE_MAIN))

    if target is not None:
        if branch == MAIN:
            if git('merge', '--ff-only', target, check=False).returncode != 0:
                raise SyncError('Updating {0} failed (local edits in the way?). Nothing was pushed.'
                                .format(MAIN))
        else:
            git('update-ref', 'refs/heads/' + MAIN, target)
    if branch != MAIN and not is_ancestor(MAIN, 'HEAD'):
        print('Merging {0} into {1}...'.format(MAIN, branch), flush=True)
        if git('merge', '--no-edit', MAIN, check=False).returncode != 0:
            git('merge', '--abort', check=False)
            raise SyncError('Merging {0} into {1} failed (a conflict with this computer\'s own '
                            'patches, or unsaved local edits in the way). Your data is committed '
                            'on {0} but nothing was pushed. Resolve it by hand, then run again.'
                            .format(MAIN, branch))


def sync():
    branch = git_out('rev-parse', '--abbrev-ref', 'HEAD')
    if branch == 'HEAD':
        raise SyncError('Detached HEAD -- check out main or this computer\'s branch first.')

    build_log()
    print(commit_data_locally(), flush=True)

    print('Fetching latest changes...', flush=True)
    git('fetch', 'origin')
    reconcile_with_origin(branch)

    if git_out('rev-parse', MAIN) != git_out('rev-parse', REMOTE_MAIN):
        print('Pushing {0}...'.format(MAIN), flush=True)
        if git('push', 'origin', '{0}:refs/heads/{0}'.format(MAIN), check=False).returncode != 0:
            raise SyncError('git push to {0} failed -- another computer probably pushed in the '
                            'meantime. Your data is committed locally; just run this again.'
                            .format(MAIN))
    else:
        print('Nothing new to push.')

    if branch != MAIN and git('push', '-u', 'origin', branch, check=False).returncode != 0:
        print('\nWARNING: backing up branch "{0}" to origin failed -- the data on {1} is '
              'unaffected.'.format(branch, MAIN))
    print('\nDone.')


if __name__ == '__main__':
    try:
        sync()
    except SyncError as e:
        print('\nERROR: {0}'.format(e))
        sys.exit(1)
