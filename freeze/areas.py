"""The archive plan: every folder of the project, where it goes, and why (single source of truth for the freeze scripts).

Repos (Hugging Face; names set by the HF_USER / repo variables in COMMANDS.md):
  results  public dataset repo  : every result folder, every Slurm log, checksum manifests, the PolyFEM binary
  data     dataset repo         : our simulated trajectories, generations, renders (derived from PhysCtrl's objects
                                  and weights; PhysCtrl declares no licence, see README "Licences")
  ckpt     model repo           : our fine-tuned checkpoints (4g fc_base_v1/v2, 14a), derived from PhysCtrl's weights
  private  ALWAYS private: rater answer codes, the redacted Claude session record (a private GitHub repo,
           COMMANDS.md step 6b; stage_hf.py still builds an HF tree for it if private HF space becomes available)
  github   the code repository  : everything else that is small (code, configs, slurm, notes, paper, PLAN, freeze/)
Not uploaded, pinned by revision + sha256 in the manifests (third party; fetch_third_party.sh restores them):
  PhysCtrl checkout + weights + test set + sample set, renderer weights (cogshader5B, LGM, SAM), HF models (FLUX.1-schnell,
  Qwen2-VL-7B, SV3D).
Excluded on purpose (never uploaded anywhere): data/hf_cache (holds a Hugging Face login token), the PokeFlex link and
its download script (personal token; PokeFlex forbids redistribution), caches (warp kernels, tmp, .cache) and the
PolyFEM source/build tree (rebuilt from commit 008a1aea with freeze/polyfem/build_polyfem.sh; the binary is uploaded).
"""
from __future__ import annotations

import os
from pathlib import Path

# defaults = the freeze-time layout; on another machine set PROJECT_ROOT / DATA_ROOT / BIG_ROOT (4/slurm/cluster.env)
PROJECT = Path(os.environ.get("PROJECT_ROOT", "/export/home/rakh90100/cvpr_physics_plausibility/4"))
WS = PROJECT.parent
DATA = Path(os.environ.get("DATA_ROOT", "/export/home/rakh90100/cvpr_physics_plausibility/data"))
BIG = Path(os.environ.get("BIG_ROOT", "/export/hbku-research/rakh90100/somw"))
FREEZE = WS / "freeze"
PRIVATE_STAGE = BIG / "freeze_private"          # redacted transcript etc. are written here before upload

# (area, source, repo, path in repo, exclude globs relative to source)
UPLOAD = [
    ("results",            PROJECT / "results",                 "results", "results", ["p11_study_codes/**"]),
    ("logs",               PROJECT / "logs",                    "results", "logs", []),
    ("polyfem_bin",        BIG / "tools/polyfem/build_conda/PolyFEM_bin", "results", "tools/polyfem/PolyFEM_bin", []),
    ("data_calib",         DATA / "calib",                      "data", "data/calib", []),
    ("data_fc_train",      DATA / "fc_train",                   "data", "data/fc_train", []),
    ("data_fresh4m",       DATA / "fresh4m",                    "data", "data/fresh4m", []),
    ("data_generations",   DATA / "generations",                "data", "data/generations", []),
    ("data_ood",           DATA / "ood",                        "data", "data/ood", []),
    ("data_coarse_traj",   DATA / "coarse_traj",                "data", "data/coarse_traj", []),
    ("data_dry_run",       DATA / "dry_run_results",            "data", "data/dry_run_results", []),
    ("big_datasets",       BIG / "datasets",                    "data", "big/datasets", []),
    ("big_generations",    BIG / "generations",                 "data", "big/generations", []),
    ("big_generations_p14a", BIG / "generations_p14a",          "data", "big/generations_p14a", []),
    ("big_coarse_traj_4o", BIG / "coarse_traj_4o",              "data", "big/coarse_traj_4o", []),
    ("big_p7b_traj",       BIG / "p7b_traj",                    "data", "big/p7b_traj", []),
    ("big_p10b_traj",      BIG / "p10b_traj",                   "data", "big/p10b_traj", []),
    ("big_p10c_traj",      BIG / "p10c_traj",                   "data", "big/p10c_traj", []),
    ("big_p10d",           BIG / "p10d",                        "data", "big/p10d", []),
    ("big_p11_examples",   BIG / "p11_examples",                "data", "big/p11_examples", []),
    ("big_p12_traj",       BIG / "p12_traj",                    "data", "big/p12_traj", []),
    ("big_renders",        BIG / "renders",                     "data", "big/renders", []),
    ("big_renders_p11",    BIG / "renders_p11",                 "data", "big/renders_p11", []),
    ("big_supp_videos",    BIG / "supp_videos",                 "data", "big/supp_videos", []),
    ("ckpt_fc_base_v1",    DATA / "checkpoints/fc_base_v1",     "ckpt", "fc_base_v1", []),
    ("ckpt_fc_base_v2",    DATA / "checkpoints/fc_base_v2",     "ckpt", "fc_base_v2", []),
    ("ckpt_p14a_ACT_F",    BIG / "checkpoints/p14a_ACT_F",      "ckpt", "p14a_ACT_F", []),
    ("ckpt_p14a_ACT_L",    BIG / "checkpoints/p14a_ACT_L",      "ckpt", "p14a_ACT_L", []),
    ("ckpt_p14a_RND_F",    BIG / "checkpoints/p14a_RND_F",      "ckpt", "p14a_RND_F", []),
    ("ckpt_p14a_RND_L",    BIG / "checkpoints/p14a_RND_L",      "ckpt", "p14a_RND_L", []),
    ("private_codes",      PROJECT / "results/p11_study_codes", "private", "p11_study_codes", []),
    ("private_session",    PRIVATE_STAGE / "claude_session",    "private", "claude_session", []),
]

# third-party bytes the results depend on: hashed into the manifest, NOT uploaded (fetch_third_party.sh)
THIRD_PARTY = [
    ("tp_physctrl_code",   DATA / "physctrl",                   [".git/**"]),
    ("tp_physctrl_ckpt",   DATA / "checkpoints/checkpoints",    []),
    ("tp_physctrl_testset", DATA / "testset",                   []),
    ("tp_physctrl_sample", DATA / "sample",                     []),
    ("tp_physctrl_zips",   DATA / "zips",                       [".cache/**"]),
    ("tp_render_ckpt",     BIG / "checkpoints/checkpoints",     []),
    ("tp_hf_models",       BIG / "hf_cache/hub",                []),
]

# never uploaded; listed so that the completeness check knows they are deliberate
EXCLUDED = {
    DATA / "hf_cache": "contains a Hugging Face login token",
    DATA / "warp_cache": "Warp kernel compile cache (rebuilt automatically)",
    DATA / "checkpoints/.cache": "huggingface-cli download cache",
    BIG / "checkpoints/.cache": "huggingface-cli download cache",
    DATA / "physctrl_commit.txt": "copied into the GitHub repo (freeze/third_party/)",
    DATA / "physctrl_requirements_somw.txt": "copied into the GitHub repo (freeze/third_party/)",
    DATA / "torch_constraints.txt": "copied into the GitHub repo (freeze/third_party/)",
    BIG / "tmp": "scratch of finished jobs",
    BIG / "tools": "PolyFEM source + build trees (binary uploaded; rebuild recipe in freeze/polyfem)",
    BIG / "hf_stage": "the symlink staging tree made by stage_hf.py",
    BIG / "freeze_private": "staging for the private repo (uploaded from there)",
    BIG / "freeze_verify": "scratch copy used by verify_numbers.py (its report is freeze/VERIFY.md)",
    BIG / "freeze_verify.log": "log of verify_numbers.py",
    BIG / "hf_upload_data.log": "log of the data upload",
    BIG / "hf_cache": "third-party model cache; its hub/ is hashed as tp_hf_models, the rest is cache",
}


def covered_roots():
    return [a[1] for a in UPLOAD] + [t[1] for t in THIRD_PARTY] + list(EXCLUDED)
