# Repository migration guide (2026-07-10)

This document is for anyone updating an existing clone of this repository, for
example the GCP VM at `~/adaptive-latent-reasoning`. The repository was reorganized
and **its entire git history was rewritten**, so every commit SHA changed and old
local branches no longer match the remote. Follow this guide to update **without
losing any local checkpoints, results, or environments**.

## What changed

1. **Branch layout.** The repo now has exactly five branches:

   | Branch | Content |
   |--------|---------|
   | `main` | Everything integrated: PonderNet codebase + `k-classifier/` + `adaptive-vectors/` + full experiment docs |
   | `option-a-k-classifier` | Final state of the upfront k* classifier approach |
   | `option-b-adaptive-vectors` | Final state of the adaptive-vectors-per-step approach |
   | `option-c-pondernet` | Final state of the PonderNet adaptive-halting approach |
   | `paper` | The written report (ACL format) |

   Old branches were deleted. Mapping from old to new:

   | Old branch | Where its content lives now |
   |------------|------------------------------|
   | `option-a`, `option-a-codi-multioutput-k-classifier` | `option-a-k-classifier` (and `k-classifier/` on `main`) |
   | `option-b` | `option-b-adaptive-vectors` (and `adaptive-vectors/` on `main`) |
   | `experiment/10-fromscratch-gpt2` | `option-c-pondernet` (its direct descendant) and `main` |
   | `experiment/{adaptive-prior,trunc-k,fullscope-prior,gamma-frontier}` | already contained in `option-c-pondernet`; documented in `docs/experiments/05..08` |
   | `pondernet`, `feat/adaptive-k-from-scratch` | unique docs integrated as `docs/experiments/09` and `docs/experiments/11` on `option-c-pondernet`/`main` |
   | `fix/validation`, `feat/pondernet-joint-training`, `fix/freeze-halting-head-only` | contained in / superseded by `option-c-pondernet` |
   | `informe` | `paper` |

2. **Folder names.** The old bare `Option-A/` and `Option-B/` folders are now named
   after what they do: `k-classifier/` and `adaptive-vectors/`, matching the
   lowercase-hyphenated style already used by `pondernet/`.

3. **History rewrite.** Commit messages were translated to English or renamed where
   they were not representative, AI co-author trailers were removed, and author
   identities were normalized (one name/email per person). The code and docs at each
   branch tip are what matters; old SHAs are gone.

4. **Docs.** All READMEs and internal docs are now in English. `docs/experiments.md`
   on `main` is the experiment index (01-11). `CLAUDE.md` was renamed to `AGENTS.md`
   (same content, tool-neutral name; several other tools read `AGENTS.md` too).

5. **Local automation folder.** The `.claude/` folder (hooks and scaffolding helpers)
   is no longer tracked in git and was removed from history. If your clone still has
   it on disk, it is now gitignored, not deleted, and the steps below preserve it. See
   "Local automation folder" further down.

## How to update a clone WITHOUT losing checkpoints

Checkpoints, eval outputs, datasets, and venvs are **gitignored**, so git will not
touch them as long as you follow two rules:

- **NEVER run `git clean -x` or `git clean -d`** (in any form: `-xdf`, `-fdx`, etc.).
  That is the only git operation here that deletes gitignored/untracked files.
- **Check for locally modified TRACKED files before any `git reset --hard`.** A hard
  reset overwrites tracked files. In particular, on the VM the old `option-a` branch
  tracked some result files (for example `Option-A/results/k_sweep_summary.csv`) that
  may have uncommitted local updates worth keeping.

Gitignored artifact locations that must survive (do not move or delete them):

```
models/pretrained/        # downloaded backbones + decoder
models/checkpoints/       # trained runs (<NN-exp>/<run-id>/ and flat legacy paths)
outputs/                  # training logs, TensorBoard events
results/                  # eval outputs
data/                     # datasets (test.jsonl IS tracked; the rest is not)
.venv*, .venv-option-a/   # virtual environments
k-classifier/models/      # downloaded weights (was Option-A/models/)
k-classifier/data/, k-classifier/results/  # partially tracked: check git status first
.claude/                  # local automation folder (see below); now gitignored
```

### Step by step

```bash
cd ~/adaptive-latent-reasoning

# 1. See what is locally modified or untracked. Anything listed as "modified" that
#    you care about: copy it aside or commit it to a local rescue branch first.
git status

# 2. Safety net: snapshot any local tracked changes on a rescue branch (cheap, local only)
git add -A && git commit -m "local snapshot before migration" || true
git branch rescue/pre-migration-$(date +%Y%m%d) || true

# 3. Back up the local automation folder if you have one; it is no longer tracked
#    upstream, and the branch switch below would otherwise delete it since it is
#    currently a TRACKED path on your old branch.
[ -d .claude ] && cp -r .claude /tmp/claude-folder-backup-$(date +%Y%m%d)

# 4. Fetch the new history and prune deleted branches
git fetch origin --prune

# 5. Move to the new main (force-recreate the local branch on the new history)
git checkout -B main origin/main

# 6. Restore the automation folder if you use it; it is now gitignored so it will
#    not be re-tracked or show up as noise in git status.
[ -d /tmp/claude-folder-backup-* ] && cp -rn /tmp/claude-folder-backup-*/. .claude/ 2>/dev/null || true

# 7. Recreate any other branch you need the same way, e.g.:
git checkout -B option-c-pondernet origin/option-c-pondernet
[ -d /tmp/claude-folder-backup-* ] && cp -rn /tmp/claude-folder-backup-*/. .claude/ 2>/dev/null || true

# 8. Delete stale local branches once you confirmed nothing local is lost
#    (list them with: git branch -vv | grep ': gone')
git branch -D <old-branch> ...

# 9. Verify the artifacts are still there
ls models/checkpoints outputs results 2>/dev/null
```

If step 1 showed local commits that are not on the new remote (work done on the VM and
never pushed), do NOT try to rebase them onto the new history wholesale. Cherry-pick
them onto the new branch instead:

```bash
git log --oneline rescue/pre-migration-*   # find the commits
git cherry-pick <sha>...                   # apply onto the new branch
```

Conflicts there are expected to be small (the rewrite changed messages, authors, and a
couple of folder names, not file contents, except for the translated Markdown docs).

### Local automation folder

If your working copy has a `.claude/` folder with local hooks or shortcuts you rely on
(for example something that stops training/eval scripts from running without an
`EXP`/`RUN` set), that folder is intentionally not part of the published history
anymore, but it is not meant to be deleted from your machine either. `.gitignore` now
excludes it, so once you restore it (step 3/6 above) it stays local, keeps working,
and will never show up as something to commit or push.

### Notes for the VM specifically

- The VM historically worked on the `option-a` branch. That branch no longer exists on
  the remote; its content (plus one extra commit with the CODI multi-output
  classifier) is now `option-a-k-classifier`, and `k-classifier/` also exists on
  `main` (renamed from `Option-A/`).
- `.venv-option-a/`, `k-classifier/models/`, and generated jsonl data are gitignored
  and live on the persistent disk: the update does not touch them.
- If a worktree exists (for example `~/alr-anapaula-optionb`), migrate or remove it
  separately (`git worktree list`); do not delete its symlinked data targets.
- After migrating, `k-classifier/AGENTS.md` (in English now, renamed from
  `Option-A/CLAUDE.md`) still documents the venv recipe and environment gotchas.

## Remotes

Both repositories were rewritten identically and hold the same five branches:

- `git@github.com:famatodlr/adaptive-latent-reasoning.git` (canonical)
- `git@github.com:anatissera/pondernet-adaptive-latent-reasoning.git` (mirror)
