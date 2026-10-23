# Global–Local Fisher Vector Fusion for Cattle Face Re-Identification

Code, frozen artefacts and result files accompanying the manuscript

> **Global–Local Fisher Vector Fusion for Cattle Face Re-Identification: A Controlled Evaluation Across Enrollment Coverage**
> Baipuhan Lü, Wei Liang\*
> School of Mechanical Engineering, Changchun University of Science and Technology, Changchun, China

This repository contains everything needed to re-derive the reported numbers, except the source images of
the two public datasets (see **Data** below).

**Archived releases** — code and frozen artefacts: [10.5281/zenodo.23557711](https://doi.org/10.5281/zenodo.23557711) (v1.0.4; concept DOI 10.5281/zenodo.23531193) ·
528 farm frames: [10.5281/zenodo.23529142](https://doi.org/10.5281/zenodo.23529142)

---

## What is here

| Path | Contents |
|---|---|
| `code/` | The evaluation pipeline exactly as frozen for the paper: global training, Hessian-affine/HardNet description, Fisher-Vector encoding, scoring, H2 evaluation, bootstrap inference, and the six scripts used for the post-hoc audits (absolute metrics, ablations, absolute CIs, farm detector diagnosis, the farm strata reconciliation, and the farm region-of-interest rule comparison). |
| `configs/` | Frozen Fisher-Vector codebook (PCA-32 / GMM-16, `whiten=False`, 1024-D) and its provenance metadata. |
| `splits/` | The custom 56/8/16 identity partition of the filtered Holstein2025 subset, and the ten frozen gallery realizations. |
| `models/` | Three frozen MobileNetV2-ArcFace global checkpoints (seeds 42/43/44, 128-D, 112x112) with their configs, and the EfficientNetV2-S arm (training records, per-epoch history and the holdout embeddings for seeds 42/43/44; the 82 MB weights are regenerable with `code/stage_e_train_global_effnetv2.py`). |
| `results/` | Machine-readable outputs behind the tables and figures (absolute metrics, ablations, bootstrap CIs, the H2025 cross-split duplicate audit with its visual review sheet, the farm evaluation of the **deployed** configuration, the same evaluation under the detector configuration kept for diagnosis, the detector failure-mode diagnosis, the farm strata reconciliation, the farm region-of-interest rule comparison, the measured pipeline cost, the PCA/GMM sensitivity, the EfficientNetV2-S backbone arm, the SIFT + Fisher Vector baseline, the comparator results and the consolidated `results_summary.csv`). |
| `manifests/` | `h2025_clean_manifest.csv` - per-image manifest of the filtered H2025 subset (identity, capture date inferred from the source filename, gallery/query role, SHA-256, source clip) - and `farm_manifest_528.csv` - per-frame manifest for the 528 farm frames (identity, session, gallery/query role, frame index, ROI flags). |
| `third_party/` | Vendored HesAffNet/HardNet++ PyTorch code (see provenance note inside). Not written by the authors. |
| `artifacts/` | `farm_frames_528.zip` — the 528 farm frames plus the manifest. **Not tracked by git**; archived on Zenodo as three part archives, see **Data** below. |

---

## Environment

```bash
conda create -n glfv python=3.10
conda activate glfv
pip install -r requirements.txt      # torch 2.7.1+cu118, ultralytics 8.4.157, ...
```

The paper's runs used PyTorch 2.7.1+cu118 on a single NVIDIA GPU, with TF32 disabled and
deterministic settings; all seeds are reported in the scripts (`42/43/44` for the global branch,
`20260917` for the bootstrap).

---

## Reproducing the main table

The pipeline expects the directory layout used during the study (`ROOT`, `DATA_ROOT` in
`code/h2025_common.py`, and the feature files produced by stages H and I). See `docs/PORTING.md`
for the three constants that must be pointed at your local copy.

```bash
# 1. descriptors -> Fisher Vectors -> scores -> paired bootstrap contrast
python code/stage_h_local_descriptors.py --subset holdout_gallery
python code/stage_i_fv_encode.py --encode holdout_query --role primary
python code/h2_evaluation_run.py

# 2. absolute Rank-1 / Macro-Rank-1 / MRR for Global, Local and Fusion
python code/reproduce_absolute_r1.py      # -> results/absolute_r1_from_frozen.json
python code/absolute_r1_ci.py             # -> identity-cluster bootstrap CIs

# 3. ablations (FV necessity, alpha sweep, backbone dependence, coverage stability)
python code/ablation_from_frozen.py

# 4. farm detector failure-mode diagnosis
python code/farm_detector_diagnosis.py

# 5. farm region-of-interest rules: comparison + consistency checks
python code/farm_roi_rule_comparison.py
```

Each script prints its numbers and writes a JSON file; the values in the manuscript can be read
directly from `results/`.

---

## Data

* **Holstein2025** — the analysed subset is derived from the publicly released Holstein2025 dataset of
  Shu et al. (2026). We redistribute the split manifests, codebook and derived features, **not** the source
  images; please obtain those from the original release.
* **CFF** — Cows Frontal Face Dataset, Zenodo, DOI [10.5281/zenodo.10535934](https://doi.org/10.5281/zenodo.10535934).
  Not mirrored here.
* **Farm frames** — 528 frames extracted from 44 clips recorded in a single session at a cattle farm in Jilin
  Province, China, with one manually verified identity per clip. Archived on Zenodo as three part archives,
  DOI [10.5281/zenodo.23529142](https://doi.org/10.5281/zenodo.23529142) (see `DATA_LICENSE.md`); the same
  frames are also bundled here as `artifacts/farm_frames_528.zip`, which is not tracked by git. The clips were
  recorded with the farm owner's permission.

---

## Licences

* **Code** (`code/`, `configs/`, `splits/`, `models/`, `results/`): MIT — see `LICENSE`.
* **Farm frames and manifests**: CC BY-NC 4.0 — see `DATA_LICENSE.md`.
* **`third_party/`**: governed by the upstream licence of the Norppa reference implementation; see
  `third_party/local_fv/README_PROVENANCE.md`. Do not relicense it under MIT.

---

## Scope and honest limitations

* The H2025 numbers come from a **custom filtered re-analysis** of a public release, not from a fully
  confirmatory, pre-registered analysis. Absolute metrics in `results/` were recomputed from the frozen
  per-query ranks and reproduce the published contrasts exactly.
* The farm evaluation comes from a **single acquisition session**: gallery and query frames of an identity
  come from the same recording, so the contrast measures within-clip discrimination. The deployed
  configuration uses a fixed bottom-centre region of interest and therefore carries **no face detector**;
  `results/farm_roi_rule_comparison.json` reports the two detector-based alternatives we measured on the
  same material, together with the checks that make the diagnostic numbers verifiable.
* No on-device latency, energy or thermal measurements were made.

---

## Citation

See `CITATION.cff`. If you use the farm frames, please also cite the dataset record (DOI assigned on
publication).

## Changes in v1.0.6

* **Measured cost of the deployed pipeline** (`results/stage_c_efficiency/`): one timing program on one machine, 20 timed repeats after warm-up, cold start reported separately, and a re-computation check that reproduces the frozen Fisher vectors bit-for-bit (max absolute difference 0.0).
* **PCA/GMM sensitivity** (`results/stage_d_summary.json`): four codebook configurations fitted on the training pool only, evaluated with the frozen global branch, gallery realizations and statistics; the first configuration reproduces the published contrast exactly, which validates the chain.
* **EfficientNetV2-S backbone arm** (`results/stage_e_effnetv2_h2025.json`): a third global branch trained under the same protocol, with the BatchNorm statistic-group deviation recorded in the run configuration.
* **SIFT + Fisher Vector baseline** (`results/stage_f_sift_h2025.json`): a traditional local descriptor under the identical descriptor budget, codebook configuration and evaluation chain.
* **Comparator reproducibility** (`code/p5_comparator/`, `results/P5C_*`): the scripts and per-query outcomes behind the same-protocol comparison table.
* **Runnability**: the vendored local-feature code is now located in `third_party/local_fv`; the Hessian-affine weights are not redistributed (upstream licence) and `code/LOCAL_WEIGHTS.md` gives their provenance and expected hashes; the two scripts with a hard-coded project root now read `H2025_ROOT` from the environment; `timm` was added to the requirements.

## Changes in v1.0.7

* **CFF arm for the EfficientNetV2-S backbone** (`results/P4C_EFFNETV2_*`): the third global representation evaluated under the CFF protocol, trained on the same nested split with the same ArcFace objective and the same inner-validation selection rule; the local branch, the region of interest and the per-fold codebooks are the frozen ones. The BatchNorm statistic-group deviation is recorded in the training record.
* **CFF coverage-ladder intervals** (`results/cff_ladder_ci/`): the identity-cluster bootstrap that the ladder plan declared and that was never executed has now been run; the three-image rung reproduces the sealed primary contrast exactly, which validates the recomputation.
* **H2025 per-query ranks** (`results/h2025_per_query/`): the per-query ranks of the global and fused arms for every regime, realization and seed, so the primary numbers can be recomputed from a file instead of by re-running the evaluation program; the accompanying validation file recomputes the reported contrast and interval from that file.
* The scripts that produced these files are in `code/`.

## Changes in v1.0.7

* **CFF arm for the EfficientNetV2-S backbone** (`results/P4C_EFFNETV2_*`): the third global representation evaluated under the CFF protocol, trained on the same nested split with the same ArcFace objective and the same inner-validation selection rule; the local branch, the region of interest and the per-fold codebooks are the frozen ones. The BatchNorm statistic-group deviation is recorded in the training record.
* **CFF coverage-ladder intervals** (`results/cff_ladder_ci/`): the identity-cluster bootstrap that the ladder plan declared and that was never executed has now been run; the three-image rung reproduces the sealed primary contrast exactly, which validates the recomputation.
* **H2025 per-query ranks** (`results/h2025_per_query/`): the per-query ranks of the global and fused arms for every regime, realization and seed, so the primary numbers can be recomputed from a file instead of by re-running the evaluation program; the accompanying validation file recomputes the reported contrast and interval from that file.
* The scripts that produced these files are in `code/`.
