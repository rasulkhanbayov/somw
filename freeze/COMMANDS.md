# Freeze commands (run in this order; Claude ran none of the git / hf commands)

Run everything on **crimv3mgpu015** (a dev server with internet), not on panther-login (no Python there).

**Status on Nov 7, 2026.** Steps 1–4 are done or run by Claude (`freeze/prepare.log` ends with `PREPARE EXIT 0`). Nothing
has been pushed or uploaded yet, so **nothing may be deleted yet**. Step 9 refuses to delete until steps 5–8 are verified.

Setup, in every new terminal:

```bash
cd ~/cvpr_physics_plausibility
source 4/slurm/cluster.env
source ~/miniconda3/etc/profile.d/conda.sh && conda activate somw
export PYTHONNOUSERSITE=1
export GH_USER=<your-github-user>
export HF_USER=rasulkhanbayov                # the account logged in on Nov 7; change it if the archive should live elsewhere
```

## 1–4. Prepared (done by Claude; to redo, in this order)

```bash
nohup python freeze/verify_numbers.py > $BIG_ROOT/freeze_verify.log 2>&1 &   # 1. every verdict re-derived: freeze/VERIFY.md, freeze/VERIFY_NOTES.md
python freeze/redact_transcript.py                                           # 2. session record without tokens
python freeze/make_manifest.py --workers 16                                  # 3. sha256 of every file (about an hour)
python freeze/stage_hf.py --repos results,data,ckpt                          # 4. upload folders (symlinks only)
```

## 5. GitHub (about 30 MB)

`gh` is not installed on this host, so first create **two empty repositories** on github.com (no README, no licence):
`somw` (public or private, your choice) and `somw-private` (must be **private**). Then:

```bash
cd ~/cvpr_physics_plausibility
git init -b main
git config user.name  "<your name>"
git config user.email "<your e-mail>"        # for a public repo, your GitHub noreply address keeps it out of the history
git add -A
git status --short | grep -iE "pokeflex/|hf_cache|^.. data/|rgcc|prepare.log" || echo "no excluded files staged"
git commit -m "Freeze of the Simulate Only What Matters project"
git tag -a freeze-2026-11-03 -m "Project freeze; data on Hugging Face (see freeze/README.md)"
git remote add origin git@github.com:$GH_USER/somw.git      # or https://github.com/$GH_USER/somw.git
git push -u origin main --tags
```

## 6. Hugging Face, all public (about 245 GB; the data repo takes hours)

Note: uploading all this public makes the human-study page unblindable (the renders match its videos by checksum).
Collect any further raters first, if you still want them.

```bash
hf auth whoami          # logged in as rasulkhanbayov with a fine-grained token. If "repo create" answers 401/403, make a
                        # token with write access (huggingface.co/settings/tokens) and run: hf auth login

hf repo create $HF_USER/somw-results     --repo-type dataset
hf repo create $HF_USER/somw-data        --repo-type dataset
hf repo create $HF_USER/somw-checkpoints --repo-type model

hf upload-large-folder $HF_USER/somw-results     $BIG_ROOT/hf_stage/results --repo-type dataset --num-workers 8
hf upload-large-folder $HF_USER/somw-checkpoints $BIG_ROOT/hf_stage/ckpt    --repo-type model   --num-workers 8
nohup hf upload-large-folder $HF_USER/somw-data  $BIG_ROOT/hf_stage/data    --repo-type dataset --num-workers 16 \
      > $BIG_ROOT/hf_upload_data.log 2>&1 &
tail -f $BIG_ROOT/hf_upload_data.log          # Ctrl+C stops watching only
```

If an upload stops (network, rate limit, quota message), run the same line again: finished files are skipped.

### 6b. Private material to the private GitHub repo (about 31 MB; never public)

The human-study answer code (no rater consent for publication) and the redacted session record stay private.

```bash
cd ~/cvpr_physics_plausibility && python freeze/redact_transcript.py        # again, so the record is complete
mkdir -p ~/somw-private && cd ~/somw-private
tar -C $BIG_ROOT/freeze_private -cf - claude_session | zstd -19 -T0 > claude_session.tar.zst
cp -r ~/cvpr_physics_plausibility/4/results/p11_study_codes .
cp ~/cvpr_physics_plausibility/freeze/cards/private.md README.md
sha256sum claude_session.tar.zst p11_study_codes/* > SHA256SUMS
git init -b main && git add -A && git commit -m "Private material of the SOMW freeze"
git remote add origin git@github.com:$GH_USER/somw-private.git && git push -u origin main
cd ~/cvpr_physics_plausibility
```

## 7. Check every uploaded file against the manifests (after all uploads have finished; minutes)

```bash
python freeze/check_remote.py \
  --repo results=$HF_USER/somw-results:dataset \
  --repo data=$HF_USER/somw-data:dataset \
  --repo ckpt=$HF_USER/somw-checkpoints:model
# must end with: REMOTE MATCHES THE MANIFESTS  and  wrote freeze/REMOTE_OK.json
```

## 8. Record the exact Hugging Face revisions in git

```bash
python - <<'PY' > freeze/RELEASE.md
import json
r = json.load(open("freeze/REMOTE_OK.json"))
print(f"# Release record (verified {r['checked_utc']})\n\n| repo | type | revision | files verified |\n|---|---|---|---|")
for v in r["repos"].values():
    print(f"| {v['repo_id']} | {v['repo_type']} | {v['revision']} | {v['files_ok']} |")
PY
cat freeze/RELEASE.md
git add -A freeze && git commit -m "Record the verified Hugging Face revisions of the freeze" && git push
```

## 9. Free the storage (only now)

The script checks everything again live, and refuses to delete if any upload is not publicly reachable at the verified
revision, or GitHub or the private repo are not pushed.

```bash
python freeze/delete_local.py                                  # plan and guard status; deletes nothing
python freeze/delete_local.py --tier ours,scratch --yes        # our data + scratch (about 245 GB)
python freeze/delete_local.py --tier renderer --yes            # renderer + model caches + PolyFEM (about 135 GB; re-downloadable)
# Optional and not recommended: --tier physctrl (about 6 GB). No copy is in our archive; re-downloading relies on
# PhysCtrl's own repositories staying online.
```

Left in place on purpose: the small project folder (`4/`, `freeze/`), `data/hf_cache` (your Hugging Face login),
`PokeFlex/`, the conda environments, and everything else in `/export/hbku-research/rakh90100` (other projects).
To free the environments too (17 GB; rebuilt by the restore steps): `conda env remove -n somw; conda env remove -n somw-fem`.

---

## Restore and reproduce (on any machine)

```bash
git clone git@github.com:$GH_USER/somw.git cvpr_physics_plausibility && cd cvpr_physics_plausibility
# 1. environments: the original recipe, then compare with the frozen lists
#    edit 4/slurm/cluster.env (PROJECT_ROOT, DATA_ROOT, BIG_ROOT, CONDA_DIR) for the new machine first
bash 4/slurm/00_setup_env.sh                       # builds env somw exactly as in October (pins inside)
conda run -n somw pip freeze --all | diff - freeze/env/somw.pip-freeze.txt
conda create -n somw-fem --file freeze/env/somw-fem.explicit.txt && conda run -n somw-fem pip install -r freeze/env/somw-fem.pip-freeze.txt
source 4/slurm/cluster.env && conda activate somw
# 2. data
DL=$BIG_ROOT/hf_download
hf download $HF_USER/somw-results     --repo-type dataset --local-dir $DL/results
hf download $HF_USER/somw-data        --repo-type dataset --local-dir $DL/data
hf download $HF_USER/somw-checkpoints --repo-type model   --local-dir $DL/ckpt
python freeze/restore.py --download-dir $DL --repos results,data,ckpt --merge
cp -r $DL/results/manifests freeze/manifests       # if the clone has none
bash freeze/fetch_third_party.sh                   # PhysCtrl, renderer, HF models at the frozen revisions + sha256 check
python freeze/check_files.py                       # every archived file: size and sha256 → ALL FILES MATCH
# 3. only if the paths differ from the freeze machine's
python freeze/remap_paths.py --project $PROJECT_ROOT --data $DATA_ROOT --big $BIG_ROOT
# 4. every number again, compared with the frozen report
python freeze/verify_numbers.py && diff freeze/VERIFY.md $BIG_ROOT/freeze_verify/VERIFY.md
# PolyFEM (Phase 12), if needed: the binary is in somw-results (tools/polyfem/PolyFEM_bin); or rebuild:
bash freeze/polyfem/build_polyfem.sh $BIG_ROOT/tools/polyfem
```
GPU experiments re-run from `4/slurm/` (scripts in phase order, `4/INDEX.md`); each run's exact command and config are
in `results/<run>/run_meta.json`.
