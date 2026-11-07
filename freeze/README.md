# Project freeze (Nov 3, 2026): what is archived where, and how to get every number back

Project: *Simulate Only What Matters: Certified Sparse Simulator Probing for Physics-Grounded Video Generation*
(CVPR 2027 draft; frozen before submission). Start with `CLAUDE.md` (workspace log), `4/CLAUDE.md`, `4/INDEX.md` and
`4/PLAN.md` (every experiment, its pre-registration, verdict and numbers).

The step-by-step commands are in `freeze/COMMANDS.md`.

## Where everything goes

| Destination | Visibility | Contents | Size |
|---|---|---|---|
| GitHub `somw` | your choice | code, configs, Slurm scripts, notes, paper source + PDF, PLAN/INDEX/CLAUDE logs, every `results/*/SUMMARY.md` and `run_meta.json`, exported decks and study page (`4/artifacts_export/`), this `freeze/` kit with the checksum manifests and exact environments | ≈ 25 MB |
| HF dataset `somw-results` | public | all of `4/results/` (rows, verdicts, figures), all Slurm logs, manifests, environments, third-party pins, the PolyFEM binary | ≈ 0.75 GB |
| HF dataset `somw-data` | public | every simulated trajectory (HDF5), generation (NPZ), probe/re-scoring trajectory and rendered video: `data/...` → `$DATA_ROOT`, `big/...` → `$BIG_ROOT` | ≈ 240 GB |
| HF model `somw-checkpoints` | public | our fine-tuned PhysCtrl checkpoints (4g fc_base_v1/v2, 14a ×4); all negative results | ≈ 3 GB |
| private GitHub `somw-private` | always private | human-study answer code; the redacted Claude Code session record (every command and output), zstd-compressed | ≈ 31 MB |
| not uploaded, pinned | — | third-party: PhysCtrl code (commit eaebd534), weights (HF chenwang/physctrl 4b3cf613), test/sample sets (bd8d316b), renderer (cogshader5B, LGM, SAM), FLUX.1-schnell, Qwen2-VL-7B, SV3D; restored by `fetch_third_party.sh`, verified by sha256 | ≈ 140 GB |

Exact sizes and file counts per area: `freeze/manifests/SUMMARY.json` (after `make_manifest.py`). The table of
areas, their destinations and the deliberate exclusions is `freeze/areas.py` (single source of truth).

**Never uploaded:** `data/hf_cache/` (contains a Hugging Face login token), the PokeFlex link and download script
(personal token; PokeFlex's licence forbids redistribution, and no PokeFlex data was downloaded), caches, the PolyFEM
source/build trees (the binary is uploaded; `freeze/polyfem/build_polyfem.sh` rebuilds it from commit 008a1aea with the
exact recorded flags), `rgcc_guidelines.txt` (internal cluster rules) and `previous_prj_slurm_files/` (another project).

## Why "nothing is lost" holds

1. `make_manifest.py` refuses to run if any file or folder under `data/` or `$BIG_ROOT` is not assigned to an upload
   area, a pinned third-party area or a listed exclusion; it records path, size and sha256 of every archived file.
2. `stage_hf.py` stages exactly the manifest's files (it stops if a count differs) as symlinks, with no copying.
3. `check_remote.py` compares each Hugging Face repo with the manifests file by file (size + sha256 / git blob id).
4. `check_files.py` does the same after a restore, and `fetch_third_party.sh` for the third-party files.
5. Things that lived only online were exported: the two result decks (`4/artifacts_export/deck_*`, slide sources; their
   images are `results/fig_qualitative/*.png`, matched by sha256) and the human-study page (`4/artifacts_export/study_page`,
   byte-identical to the published page; its server-side store was empty, the one answer code is in `somw-private`).
6. Numbers first computed inline in the session are now scripts with stored output: `code/p14a_diagnose.py`,
   `code/p12_s0_diagnose.py` (Oct 24). Everything else inline is in the redacted session record (private repo).

## Why "every number is reproducible" holds

- **From the stored outputs (CPU, hours):** `freeze/verify_numbers.py` re-runs every verdict that was computed on a dev
  server (all `SUMMARY.md` files without a Slurm job: Phases 1, 2, 4a–4o, 5a, 7a–7d, 8a–8g, 9, 10a–10e, 11, 14a, 15a, the
  paper tables) from a copy of `results/` and the archived data, and compares each new `SUMMARY.md` and paper table with
  the stored one byte for byte. Its report (`VERIFY.md`) is part of the freeze (see COMMANDS step 1). Result: 46 of 50 identical on
  the first pass; the other 4 are traced in `VERIFY_NOTES.md` and change no reported number.
- **From scratch (GPU):** every GPU result has its exact command and config in `results/<run>/run_meta.json`
  (argv, config, host, GPU, PhysCtrl commit) and its job script in `4/slurm/` (numbered in run order; `INDEX.md` maps
  them). The environments are `freeze/env/` (conda explicit lists + pip freeze). GPU MPM is not bit-reproducible:
  re-simulated trajectories match the stored ones to about 1e-3 (measured: 2e-7 to 2.2e-3, `PLAN.md` Phases 6a, 10c),
  which is why the stored trajectories are archived and the verdicts are verified from them.

## Licences (all Hugging Face repos public, decided by the user on Nov 3, 2026)

PhysCtrl releases its code, weights and data **without a licence** (checked Nov 3, 2026: no LICENSE in the repository,
no licence on the Hugging Face model or dataset). Our trajectories use PhysCtrl's objects (from Objaverse via
TRELLIS-500K), and our generations, renders and fine-tuned checkpoints derive from PhysCtrl's weights. They are
published publicly (the user's decision; no private storage available), and the repo cards state this provenance and
that reuse of the objects and model outputs is subject to PhysCtrl's and Objaverse's terms. Everything in `somw-results`
and GitHub is our own work (numbers, code, text, figures). The rater answer code and the session record are not public
(private GitHub repo `somw-private`). Note: a public GitHub/HF release with your name breaks
double-blind anonymity if this paper is later submitted to a double-blind venue (use anonymous mirrors for review).

## Freeing storage

`freeze/delete_local.py` deletes in tiers (`ours`, `scratch`, `renderer`, `physctrl`) and refuses unless `check_remote.py`
passed, every Hugging Face repo is publicly reachable at the verified revision, and GitHub and the private repo are pushed.
Keeping the `physctrl` tier (about 6 GB) is recommended: PhysCtrl's files are not in our archive and depend on its repos.

## Restore and reproduce (summary; exact commands in COMMANDS.md, "Restore")

1. Clone GitHub; create the envs from `freeze/env/*.explicit.txt` (+ pip freeze); edit `4/slurm/cluster.env` paths.
2. Download the HF repos; `python freeze/restore.py --download-dir DL` puts each area back in place.
3. `bash freeze/fetch_third_party.sh` (pinned third-party files + sha256 check).
4. `python freeze/check_files.py` (every archived file, sha256); `python freeze/remap_paths.py` only if paths changed.
5. `python freeze/verify_numbers.py` and compare with the frozen `VERIFY.md`.
