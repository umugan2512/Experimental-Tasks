"""Pull the shared training_log.xlsx, refresh it from this box's local sessions, and push it to main.

Run via `Sync Training Log.bat`. Works the same on every box, whichever branch it has checked out:

- On `main` (the default for a shared box): plain `git pull` -> rebuild -> commit + push.
- On a box-specific branch (e.g. `training-box-5`, which carries code changes meant for that box
  only): `origin/main` is merged INTO the box branch first (so the box picks up shared code and every
  other box's latest training_log.xlsx), then the rebuilt training_log.xlsx is committed directly
  onto `origin/main` -- via a temporary index, without ever checking `main` out -- and pushed there.
  Only training_log.xlsx ever lands on main from a box branch; the box's own code never does. That
  same commit is then merged back into the box branch so the two stay in step, and the box branch
  itself is pushed to origin as a backup (it's never merged into main, so other boxes never see it).
"""
import os
import subprocess
import sys
import tempfile

RECORDS_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_NAME = 'training_log.xlsx'
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


def xlsx_repo_path():
    """training_log.xlsx's path relative to the repo root, forward-slashed (what git's index uses)."""
    return git_out('rev-parse', '--show-prefix') + XLSX_NAME


def build_log():
    print('Refreshing training_log.xlsx from local session data...', flush=True)
    rc = subprocess.call([sys.executable, 'build_training_log.py'], cwd=RECORDS_DIR)
    if rc != 0:
        raise SyncError('build_training_log.py failed -- see the message above. Nothing was committed.')


def xlsx_changed_vs(ref):
    return git('diff', '--quiet', ref, '--', XLSX_NAME, check=False).returncode != 0


def commit_message():
    return 'Update training_log.xlsx from {0}'.format(os.environ.get('COMPUTERNAME', 'unknown box'))


def sync_on_main():
    print('Pulling latest changes...', flush=True)
    if git('pull', check=False).returncode != 0:
        raise SyncError('git pull failed (often a conflict that needs resolving by hand, since '
                        'training_log.xlsx is a binary file git can\'t auto-merge). Nothing was '
                        'updated or pushed. Fix this first, then run again.')
    build_log()
    if not xlsx_changed_vs('HEAD'):
        print('\nNo changes to training_log.xlsx since the last sync -- nothing to commit.')
        return
    print('\nCommitting and pushing training_log.xlsx...', flush=True)
    git('add', XLSX_NAME)
    git('commit', '-m', commit_message())
    if git('push', check=False).returncode != 0:
        raise SyncError('git push failed -- your commit is saved locally but not shared yet. '
                        'Someone else may have pushed in the meantime -- try running this again.')
    print('\nDone -- training_log.xlsx updated, committed, and pushed.')


def commit_xlsx_onto(parent_ref):
    """Commit the working-tree training_log.xlsx on top of `parent_ref` without touching HEAD,
    the real index, or any other file. Returns the new commit's hash."""
    blob = git_out('hash-object', '-w', XLSX_NAME)
    fd, tmp_index = tempfile.mkstemp(prefix='sync_training_log_index_')
    os.close(fd)
    os.remove(tmp_index)  # git wants to create the index file itself
    try:
        env = dict(os.environ, GIT_INDEX_FILE=tmp_index)
        git('read-tree', parent_ref, env=env)
        git('update-index', '--add', '--cacheinfo', '100644,{0},{1}'.format(blob, xlsx_repo_path()),
            env=env)
        tree = git_out('write-tree', env=env)
    finally:
        if os.path.exists(tmp_index):
            os.remove(tmp_index)
    return git_out('commit-tree', tree, '-p', parent_ref, '-m', commit_message())


def sync_on_box_branch(branch):
    print('On box branch "{0}" -- training_log.xlsx goes to {1}, this box\'s code stays on "{0}".'
          .format(branch, MAIN), flush=True)
    print('Fetching latest changes...', flush=True)
    git('fetch', 'origin')
    upstream = 'origin/' + MAIN

    print('Merging {0} into {1}...'.format(upstream, branch), flush=True)
    if git('merge', '--no-edit', upstream, check=False).returncode != 0:
        git('merge', '--abort', check=False)
        raise SyncError('Merging {0} into {1} failed (a conflict between shared code and this box\'s '
                        'own changes, or unsaved local edits in the way). Nothing was updated or '
                        'pushed. Resolve it by hand, then run again.'.format(upstream, branch))

    build_log()
    if not xlsx_changed_vs(upstream):
        print('\nNo changes to training_log.xlsx since the last sync -- nothing to commit.')
    else:
        print('\nCommitting training_log.xlsx onto {0} and pushing...'.format(MAIN), flush=True)
        new_commit = commit_xlsx_onto(upstream)
        if git('push', 'origin', '{0}:refs/heads/{1}'.format(new_commit, MAIN), check=False).returncode != 0:
            raise SyncError('git push to {0} failed -- someone else probably pushed in the meantime. '
                            'Your training_log.xlsx edits are still in the working copy; just run '
                            'this again.'.format(MAIN))
        git('fetch', 'origin')
        # The working copy already matches new_commit's xlsx -- reset it so the merge can apply.
        git('checkout', 'HEAD', '--', XLSX_NAME)
        git('merge', '--no-edit', new_commit)
        # Keep the local `main` branch in step too, if it's simply behind (never force it).
        if git('merge-base', '--is-ancestor', MAIN, new_commit, check=False).returncode == 0:
            git('branch', '-f', MAIN, new_commit, check=False)
        print('training_log.xlsx pushed to {0}.'.format(MAIN))

    if git('push', '-u', 'origin', branch, check=False).returncode != 0:
        print('\nWARNING: backing up branch "{0}" to origin failed -- training_log.xlsx itself is '
              'unaffected.'.format(branch))
    print('\nDone.')


def main():
    branch = git_out('rev-parse', '--abbrev-ref', 'HEAD')
    if branch == 'HEAD':
        raise SyncError('Detached HEAD -- check out main or this box\'s branch first.')
    if branch == MAIN:
        sync_on_main()
    else:
        sync_on_box_branch(branch)


if __name__ == '__main__':
    try:
        main()
    except SyncError as e:
        print('\nERROR: {0}'.format(e))
        sys.exit(1)
