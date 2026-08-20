# Masking literature — papers and their skills

Five selected open-access papers on detector masking / outlier rejection for X-ray area detectors,
each with a companion method file in [`skills/masking/`](../../skills/masking/README.md)
recording what the method prescribes and how (or whether) this repo implements it.

| PDF | Paper | Skill | Status here |
|---|---|---|---|
| `05_Sadri_2022_Robust_Mask_Maker_JApplCryst.pdf` | Sadri *et al.* (2022), *J. Appl. Cryst.* **55**, 1549 — Automatic bad-pixel mask maker (RMM) | [09](../../skills/masking/09_rmm_robust_mask_maker.md) | implemented as methods 03/04 |
| `01_Barty_2014_Cheetah_JApplCryst.pdf` | Barty *et al.* (2014), *J. Appl. Cryst.* **47**, 1118 — Cheetah | [10](../../skills/masking/10_cheetah_mask_layers.md) | layer discipline followed; dynamic layers not wired |
| `08_Kieffer_2025_Signal_Separation_Sigma_Clipping.pdf` | Kieffer *et al.* (2025), *J. Appl. Cryst.* **58**, 138 (arXiv:2411.09515) — signal separation / azimuthal sigma-clipping | [11](../../skills/masking/11_azimuthal_sigma_clip_signal_separation.md) | implemented as method 02 |
| `06_Hadian_Jazi_Sadri_2023_Robust_Statistics_Python.pdf` | Hadian-Jazi & Sadri (2023), *Acta Cryst.* **D79**, 820 — RGFlib robust statistics | [12](../../skills/masking/12_rgflib_robust_statistics.md) | estimators reimplemented, library not a dependency |
| `07_Yanxon_2023_UNet_XRD_Segmentation.pdf` | Yanxon *et al.* (2023), arXiv:2310.16186 (APS/ALS) — U-Net artifact segmentation | [13](../../skills/masking/13_unet_artifact_segmentation.md) | not applied (no labelled corpus) |

The folder also contains three background references on psana and detector-artifact
handling. All eight PDFs are local research inputs and are intentionally ignored by Git;
this index records the curated sources without redistributing the files.

All five selected papers are open access (PMC / arXiv); PDFs downloaded 2026-08-04. Two of them
(01, 04) are the published origin of methods already wired into this pipeline, one (03) is
the origin of the pyFAI clipping used in method 02, one (02) is the production practice this
repo's layer accounting follows, and one (05) is the learned option this campaign has not
adopted.
