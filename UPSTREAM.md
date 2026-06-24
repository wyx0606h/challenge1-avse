# Upstream baseline maintenance

This repository keeps the official Real-World AVSE baseline history and adds
participant-facing Track 2 reproducibility tooling on top.

## Remotes

```text
origin    https://github.com/wyx0606h/challenge1-avse.git
upstream  https://github.com/Real-World-AVSE/Baseline.git
```

The baseline imported for the initial Track 2 bootstrap is:

```text
1d244637097bd2b9f2e44b96b973d2f7bbf86291
```

## Local changes

The initial participant-maintained surface is intentionally small:

- `docs/TRACK2_CHALLENGE_GUIDE_ZH.md`
- `tools/check_track2_setup.py`
- `scripts/smoke_track2.sh`
- `UPSTREAM.md`
- additional repository-local ignore rules in `.gitignore`

The official `README.md`, `LICENSE`, model implementation, training
configuration, checkpoint format, and evaluation implementation are kept
unchanged in the bootstrap pull request.

## Syncing future upstream changes

Create a dedicated sync branch instead of merging directly into `main`:

```bash
git fetch upstream
git checkout main
git pull --ff-only origin main
git checkout -b codex/sync-upstream-YYYYMMDD
git merge --no-ff upstream/main
```

Resolve conflicts by preserving upstream behavior first, then reapply the small
participant-facing additions above. Run the checks described in
`docs/TRACK2_CHALLENGE_GUIDE_ZH.md` before opening a pull request.

Do not push local datasets, pretrained weights, enhanced waveforms, experiment
directories, enrollment embeddings, or organizer-provided test assets.
