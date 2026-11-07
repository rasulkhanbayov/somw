# Re-derivation of every stored verdict (freeze/verify_numbers.py)

Work folder: /export/hbku-research/rakh90100/somw/freeze_verify

| run | command | result | seconds |
|---|---|---|---|
| p12_s0 | — | not re-run: the S0 gate itself (PolyFEM, hours on cpu-all: slurm/102_p12_s0_cpu.sh) | |
| p1_summary | `code/p1_summarize.py --config configs/p1_summary.yaml` | IDENTICAL | 4 |
| p2_summary | `code/p2_summarize.py --config configs/p2_summary.yaml` | IDENTICAL | 173 |
| p2_summary_v2 | `code/p2_summarize.py --config configs/p2_summary_v2.yaml` | IDENTICAL | 196 |
| p2c_summary | `code/p2c_summarize.py --config configs/p2c_summary.yaml` | IDENTICAL | 238 |
| p4a_certificate | `code/p4a_certificate.py --config configs/p4a_certificate.yaml` | IDENTICAL | 107 |
| p4b_summary | `code/p4b_summarize.py --config configs/p4b_repair.yaml` | IDENTICAL | 5 |
| p4b_certificate_objects | `code/p4b_certificate_objects.py --config configs/p4b_certificate_objects.yaml` | IDENTICAL | 341 |
| p4c_summary | `code/p4c_summarize.py --config configs/p4c_repair.yaml` | IDENTICAL | 7 |
| p4e_summary | `code/p4e_summarize.py --config configs/p4e_repair.yaml` | IDENTICAL | 4 |
| p4d_pipeline | `code/p4d_pipeline.py --config configs/p4d_pipeline.yaml` | IDENTICAL | 11 |
| p4f_confirm | `code/p4f_confirm_pipeline.py --config configs/p4f_confirm.yaml` | IDENTICAL | 171 |
| p4g_summary | `code/p4g_summarize.py --config configs/p4g_repair.yaml` | DIFFERS (1 changed lines; logs/p4g_summary.diff) | 4 |
| p4g_summary_v2 | `code/p4g_summarize.py --config configs/p4g_repair_v2.yaml` | IDENTICAL | 4 |
| p4h_summary | `code/p4h_summarize.py --config configs/p4h_suffix.yaml` | IDENTICAL | 103 |
| p4i_summary | `code/p4i_summarize.py --config configs/p4i_state.yaml` | IDENTICAL | 11 |
| p4j_confirm | `code/p4j_confirm.py --config configs/p4j_confirm.yaml` | IDENTICAL | 103 |
| p4k_gated | `code/p4k_gated.py --config configs/p4k_gated.yaml` | IDENTICAL | 164 |
| p4m_confirm | `code/p4m_confirm.py --config configs/p4m_confirm.yaml` | IDENTICAL | 212 |
| p4l_summary | `code/p4l_summarize.py --config configs/p4l_summary.yaml` | IDENTICAL | 630 |
| p4l_summary_ref | `code/p4l_summarize.py --config configs/p4l_summary_ref.yaml` | IDENTICAL | 139 |
| p4n_summary | `code/p4n_summarize.py --config configs/p4n_summary.yaml` | IDENTICAL | 654 |
| p4o_summary | `code/p4n_summarize.py --config configs/p4o_summary.yaml` | IDENTICAL | 652 |
| p5a_amortise | `code/p5a_amortise.py --config configs/p5a_amortise.yaml` | IDENTICAL | 1087 |
| p7a_metrics | `code/p7a_metrics.py --config configs/p7a_metrics.yaml` | IDENTICAL | 550 |
| p7c_targets | `code/p7c_targets.py --config configs/p7c_targets.yaml` | IDENTICAL | 146 |
| p8b_fused | `code/p8b_fused.py --config configs/p8b_fused.yaml` | IDENTICAL | 86 |
| p8a_pilot | `code/p8a_pilot.py --config configs/p8a_pilot.yaml` | IDENTICAL | 73 |
| p7d_contact | `code/p7d_contact.py --config configs/p7d_contact.yaml` | IDENTICAL | 723 |
| p8d_pilot | `code/p8d_pilot.py --config configs/p8d_pilot.yaml` | IDENTICAL | 6 |
| p8f_cheapcheck | `code/p8f_cheapcheck.py --config configs/p8f_cheapcheck.yaml` | IDENTICAL | 45 |
| p8e_stiff | `code/p8e_stiff.py --config configs/p8e_stiff.yaml` | IDENTICAL | 56 |
| p8g_stiff | `code/p8e_stiff.py --config configs/p8g_stiff.yaml` | IDENTICAL | 53 |
| p9c_summary | `code/p4n_summarize.py --config configs/p9c_summary.yaml` | IDENTICAL | 312 |
| p7b_testset_volume | `code/p7b_summarize.py --config configs/p7b_testset_volume.yaml` | IDENTICAL | 156 |
| p7b_testset | `code/p7b_summarize.py --config configs/p7b_testset.yaml` | IDENTICAL | 15 |
| p10a_convergence | `code/p10a_convergence.py --config configs/p10a_summary.yaml` | DIFFERS (4 changed lines; logs/p10a_convergence.diff) | 551 |
| p10b_testset | `code/p7b_summarize.py --config configs/p10b_testset.yaml` | IDENTICAL | 85 |
| p9_confirm | `code/p9_confirm.py --config configs/p9_confirm.yaml` | IDENTICAL | 60 |
| p10c_summary | `code/p10c_summary.py --config configs/p10c_rescore.yaml` | IDENTICAL | 323 |
| p10c_explore | `code/p10c_explore.py --config configs/p10c_rescore.yaml` | IDENTICAL | 296 |
| p9_table | `code/p9_table.py --config configs/p9_confirm.yaml --contact-config configs/p9c_summary.yaml` | IDENTICAL | 114 |
| p11_summary | `code/p11_summary.py --config configs/p11_judge.yaml` | IDENTICAL | 5 |
| p14a_acquire | `code/p14a_acquire.py` | DIFFERS (1 changed lines; logs/p14a_acquire.diff) | 593 |
| p10e_wall | `code/p10e_wall.py` | IDENTICAL | 220 |
| p14a_summary | `code/p14a_summary.py` | IDENTICAL | 303 |
| p14a2_acquire | `code/p14a_acquire.py --wall-free --run-name p14a2_acquire` | IDENTICAL | 503 |
| p15a_scenes | `code/p15_scenes.py` | IDENTICAL | 603 |
| p12_s0_diagnose | `code/p12_s0_diagnose.py` | FAILED (exit 1; see logs/p12_s0_diagnose.log) | 1 |
| p14a_diagnose | `code/p14a_diagnose.py` | IDENTICAL | 619 |
| p10d_summary | `code/p10d_strands.py --summarize` | IDENTICAL | 27 |

## Paper tables

- paper/tab/main_table.tex: IDENTICAL
- paper/tab/reference_table.tex: IDENTICAL

## Totals

- DIFFERS: 3
- FAILED: 1
- IDENTICAL: 46
- not re-run: 1
