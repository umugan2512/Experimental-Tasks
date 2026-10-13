"""Commit and share this lab's DATA (training_log.xlsx + session data) on main, from any branch.

Data = `records/training_log.xlsx` plus the session folders under `AuditoryEvidenceAccum/experiments/`
(session CSV, `_struct.json`/`.mat`, video timestamps -- `.gitignore` keeps out videos, preview
JPGs, launch logs, session-local user_settings.py, and the box-specific setup `.json` files).

Two entry points:
- `commit_data_locally()` -- called after every weight/notes Save in run_session.py. Commits the
  data onto the LOCAL `main` branch only; never pushes.
- `main()` -- run via `Sync Training Log.bat`: bring in the latest main, rebuild training_log.xlsx,
  commit the data onto main, and push main.

Works the same whichever branch is checked out:
- On `main`: the data is committed straight onto main.
- On a computer-specific branch (e.g. `training05`: main plus only that machine's own patches,
  such as interpreter paths -- all real code lives on main): main is never checked out. The data
  commit is built on main via a temporary index, and a matching merge commit (main -> this
  branch) is written by hand, so the files already on disk become tracked without git's own
  merge refusing to overwrite them. Only data ever lands on main from such a branch.

Session files deleted locally are never recorded as deletions (`git add --ignore-removal`), so a
box clearing out old local sessions can't remove other boxes' data from main.
"""
import os
import subprocess
import sys
import tempfile

RECORDS_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_NAME = 'training_log.xlsx'
# Relative to RECORDS_DIR (every git call runs from there).
DATA_PATHS = [XLSX_NAME, os.path.join('..', 'experiments')]
MAIN = 'main'


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


def is_ancestor(a, b):
    return git('merge-base', '--is-ancestor', a, b, check=False).returncode == 0


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
    untouched. Returns the tree hash."""
    fd, tmp_index = tempfile.mkstemp(prefix='sync_training_log_index_')
    os.close(fd)
    os.remove(tmp_index)  # git wants to create the index file itself
    try:
        env = dict(os.environ, GIT_INDEX_FILE=tmp_index)
        git('read-tree', base_ref, env=env)
        git('add', '--ignore-removal', '--', *DATA_PATHS, env=env)
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


def sync():
    branch = git_out('rev-parse', '--abbrev-ref', 'HEAD')
    if branch == 'HEAD':
        raise SyncError('Detached HEAD -- check out main or this computer\'s branch first.')
    print('Fetching latest changes...', flush=True)
    git('fetch', 'origin')
    remote_main = 'origin/' + MAIN

    if branch == MAIN:
        print('Pulling latest changes...', flush=True)
        if git('pull', '--no-edit', check=False).returncode != 0:
            raise SyncError('git pull failed (often a conflict that needs resolving by hand, since '
                            'training_log.xlsx is a binary file git can\'t auto-merge). Nothing was '
                            'updated or pushed. Fix this first, then run again.')
    else:
        # Bring local main up to date without checking it out. Unpushed local commits on main
        # (e.g. from run_session.py's Save) are kept and pushed below.
        if is_ancestor(MAIN, remote_main):
            git('update-ref', 'refs/heads/' + MAIN, git_out('rev-parse', remote_main))
        elif not is_ancestor(remote_main, MAIN):
            raise SyncError('Local {0} and origin/{0} have diverged (unpushed local commits AND new '
                            'commits on origin). Nothing was updated or pushed. Resolve it by hand '
                            '(e.g. git switch {0}; git pull), then run again.'.format(MAIN))
        print('Merging {0} into {1}...'.format(MAIN, branch), flush=True)
        if git('merge', '--no-edit', MAIN, check=False).returncode != 0:
            git('merge', '--abort', check=False)
            raise SyncError('Merging {0} into {1} failed (a conflict with this computer\'s own '
                            'patches, or unsaved local edits in the way). Nothing was updated or '
                            'pushed. Resolve it by hand, then run again.'.format(MAIN, branch))

    build_log()
    print('\n' + commit_data_locally(), flush=True)

    if git_out('rev-parse', MAIN) != git_out('rev-parse', remote_main):
        print('Pushing {0}...'.format(MAIN), flush=True)
        if git('push', 'origin', '{0}:refs/heads/{0}'.format(MAIN), check=False).returncode != 0:
            raise SyncError('git push to {0} failed -- someone else probably pushed in the meantime. '
                            'Your data is committed locally; just run this again.'.format(MAIN))
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
