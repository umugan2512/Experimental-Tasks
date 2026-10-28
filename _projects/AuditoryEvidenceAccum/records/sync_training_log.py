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


# Submodules whose pinned commits this lab makes itself, so they have to reach GitHub BEFORE a main
# that points at them -- otherwise every other computer's `git submodule update` fails with
# "...points to a commit that does not exist". Deepest first: each one's parent pins it.
# (pybpod/base/pybpod-api is not here: it has no lab fork and stays pinned to an official
# release.) push.recurseSubmodules=on-demand would do this too, but it reuses the superproject's
# refspec (main) inside the submodules, which have no main branch, and the push fails.
FORKED_SUBMODULES = ['pybpod/plugins/pybpod-gui-plugin-rotaryencoder', 'pybpod']
CHECKED_PINS = [('pybpod', 'pybpod'), ('pybpod', 'pybpod/base/pybpod-api'),
                ('pybpod', 'pybpod/plugins/pybpod-gui-plugin-rotaryencoder')]


def git_in(path, *args, check=True):
    """ git run inside repo_root()/path (a submodule checkout). """
    proc = subprocess.run(['git', '-C', os.path.join(repo_root(), path)] + list(args),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if check and proc.returncode != 0:
        raise SyncError('git -C {0} {1} failed:\n{2}{3}'.format(path, ' '.join(args), proc.stdout,
                                                              proc.stderr))
    return proc


def push_forked_submodules():
    """ Pushes each forked submodule's current branch if it has commits its remote lacks. """
    for path in FORKED_SUBMODULES:
        if not os.path.isdir(os.path.join(repo_root(), path)):
            continue
        branch = git_in(path, 'symbolic-ref', '-q', '--short', 'HEAD', check=False).stdout.strip()
        if not branch:
            continue   # detached: nothing of ours to push from here; pinned_commits_pushed() decides
        git_in(path, 'fetch', 'origin', check=False)
        ahead = git_in(path, 'rev-list', '--count', 'origin/{0}..HEAD'.format(branch),
                       check=False).stdout.strip()
        if ahead and ahead != '0':
            print('Pushing submodule {0} ({1}, {2} commit(s))...'.format(path, branch, ahead),
                  flush=True)
            if git_in(path, 'push', 'origin', branch, check=False).returncode != 0:
                raise SyncError('git push of submodule {0} failed. Nothing on {1} was pushed.'
                                .format(path, MAIN))


def pinned_commits_pushed(rev):
    """ Every lab-relevant gitlink that `rev` (directly, or via pybpod) pins is on its remote. """
    pins = {}
    for parent, path in CHECKED_PINS:
        if parent == path:   # pinned by this repo itself
            line = git_in('.', 'ls-tree', rev, '--', path, check=False).stdout.strip()
        else:
            sub = path[len(parent) + 1:]
            parent_pin = pins.get(parent)
            if parent_pin is None:
                continue
            line = git_in(parent, 'ls-tree', parent_pin, '--', sub, check=False).stdout.strip()
        if not line:
            continue
        commit = line.split()[2]
        pins[path] = commit
        remote = git_in(path, 'branch', '-r', '--contains', commit, check=False).stdout.strip()
        if not remote:
            raise SyncError('{0} pins {1} at {2}, which is not on its GitHub remote -- other '
                            'computers could not check it out, so {0} was not pushed.'
                            .format(MAIN, path, commit[:7]))


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


def _ref_exists(ref):
    return git('rev-parse', '--verify', '-q', ref + '^{commit}', check=False).returncode == 0


def _owned_sheet_names():
    return {build_training_log._sheet_name_for(s) for s in build_training_log.owned_subjects()}


def newest_main_xlsx():
    """(bytes, blob hash) of training_log.xlsx on the newest main this computer knows about --
    origin/main or local main, whichever contains the other; if they diverged, origin/main's
    workbook with this computer's own sheets from local main (same rule as _merge_mains()). No
    network access: run `git fetch` first for the very latest. (None, None) if main has no log."""
    path = xlsx_repo_path()
    refs = [r for r in (REMOTE_MAIN, MAIN) if _ref_exists(r)]
    if len(refs) == 2 and not is_ancestor(MAIN, REMOTE_MAIN):
        if not is_ancestor(REMOTE_MAIN, MAIN):
            theirs, ours = blob_at(REMOTE_MAIN, path), blob_at(MAIN, path)
            if theirs and ours and theirs != ours:
                merged = xlsx_sheet_merge.merge_workbooks(
                    git_bytes('cat-file', 'blob', theirs), git_bytes('cat-file', 'blob', ours),
                    _owned_sheet_names())
                return merged, None
        refs = [MAIN, REMOTE_MAIN]   # local main is newest (or blobs identical / one missing)
    for ref in refs:
        blob = blob_at(ref, path)
        if blob:
            return git_bytes('cat-file', 'blob', blob), blob
    return None, None


def xlsx_from_main():
    """The workbook build_training_log.py should start from, as bytes -- or None if the file on
    disk is already right. A computer-specific branch's copy of training_log.xlsx goes stale as
    soon as main moves ahead (and switching branches swaps the file on disk), so the newest main's
    copy is used instead. Unsaved local edits are kept: if the file on disk differs from HEAD's
    committed copy, this computer's own mice's sheets are taken from disk."""
    main_bytes, main_blob = newest_main_xlsx()
    if main_bytes is None:
        return None
    disk_path = os.path.join(RECORDS_DIR, XLSX_NAME)
    if not os.path.exists(disk_path):
        return main_bytes
    head_blob = blob_at('HEAD', xlsx_repo_path())
    if main_blob is not None and main_blob == head_blob:
        return None            # disk is already based on the newest main
    disk_blob = git_out('hash-object', '--', XLSX_NAME)
    if disk_blob == main_blob:
        return None
    if disk_blob == head_blob:  # no local edits -- just take main's
        return main_bytes
    with open(disk_path, 'rb') as f:
        return xlsx_sheet_merge.merge_workbooks(main_bytes, f.read(), _owned_sheet_names())


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
            # update-index takes the two path forms differently (confirmed on git 2.28):
            # - plain path args (`--force-remove -- <path>`) resolve relative to the process's own
            #   cwd (RECORDS_DIR), so the repo-root-relative `path` from diff-index silently matched
            #   nothing (exit 0, no effect) -- these exclusions never worked. Use a
            #   RECORDS_DIR-relative path there (DATA_PATHS[1]'s own '../experiments' form).
            # - `--cacheinfo <mode>,<sha>,<path>` takes the path as-is, relative to the repo root,
            #   and rejects '../' outright ("fatal: Invalid path"), so it keeps `path` unchanged.
            base_blob = blob_at(base_ref, path)
            if base_blob:
                git('update-index', '--cacheinfo', '100644,{0},{1}'.format(base_blob, path), env=env)
            else:
                cwd_relative = '../experiments/' + path[len(exp_prefix):]
                git('update-index', '--force-remove', '--', cwd_relative, env=env)
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
        # training_log.xlsx on disk was already rebuilt on top of main's copy (see
        # xlsx_from_main()), so set it aside for the merge and put it back afterwards --
        # otherwise git refuses to merge over the modified file.
        disk_path = os.path.join(RECORDS_DIR, XLSX_NAME)
        held = None
        if (os.path.exists(disk_path) and blob_at('HEAD', xlsx_repo_path())
                and git_out('hash-object', '--', XLSX_NAME) != blob_at('HEAD', xlsx_repo_path())):
            with open(disk_path, 'rb') as f:
                held = f.read()
            git('checkout', 'HEAD', '--', XLSX_NAME)
        merged_ok = git('merge', '--no-edit', MAIN, check=False).returncode == 0
        if not merged_ok:
            git('merge', '--abort', check=False)
        if held is not None:
            with open(disk_path, 'wb') as f:
                f.write(held)
        if not merged_ok:
            raise SyncError('"{0}" is behind {1} and merging it in failed (a conflict with this '
                            'computer\'s own patches, or other unsaved edits in the way). Nothing '
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

    # Fetch BEFORE rebuilding, so the rebuild starts from the very latest training_log.xlsx on
    # main (see xlsx_from_main()). Offline is fine here; the required fetch below reports it.
    print('Fetching latest changes...', flush=True)
    fetched = git('fetch', 'origin', check=False).returncode == 0

    build_log()
    print(commit_data_locally(), flush=True)

    if not fetched:
        git('fetch', 'origin')
    reconcile_with_origin(branch)

    if git_out('rev-parse', MAIN) != git_out('rev-parse', REMOTE_MAIN):
        push_forked_submodules()
        pinned_commits_pushed(MAIN)
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
