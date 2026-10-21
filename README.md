# Global–Local Fisher Vector Fusion for Cattle Face Re-Identification

Code, frozen artefacts and result files accompanying the manuscript

> **Global–Local Fisher Vector Fusion for Cattle Face Re-Identification: A Controlled Evaluation Across Enrollment Coverage**
> Baipuhan Lü, Wei Liang\*
> School of Mechanical Engineering, Changchun University of Science and Technology, Changchun, China

This repository contains everything needed to re-derive the reported numbers, except the source images of
the two public datasets (see **Data** below).

---

## What is here

| Path | Contents |
|---|---|
| `code/` | The evaluation pipeline exactly as frozen for the paper: global training, Hessian-affine/HardNet description, Fisher-Vector encoding, scoring, H2 evaluation, bootstrap inference, and the four scripts used for the post-hoc audits (absolute metrics, ablations, absolute CIs, farm detector diagnosis). |
| `configs/` | Frozen Fisher-Vector codebook (PCA-32 / GMM-16, `whiten=False`, 1024-D) and its provenance metadata. |
| `splits/` | The custom 56/8/16 identity partition of the filtered Holstein2025 subset, and the ten frozen gallery realizations. |
| `models/` | Three frozen MobileNetV2-ArcFace global checkpoints (seeds 42/43/44, 128-D, 112x112) with their configs. |
| `results/` | Machine-readable outputs behind the tables and figures (absolute metrics, ablations, bootstrap CIs, farm evaluation, detector diagnosis). |
| `manifests/` | Per-frame manifest for the 528 farm frames (identity, session, gallery/query role, frame index, ROI flags). |
| `third_party/` | Vendored HesAffNet/HardNet++ PyTorch code (see provenance note inside). Not written by the authors. |
| `artifacts/` | `farm_frames_528.zip` — the 528 farm frames plus the manifest. **Not tracked by git**; upload to Zenodo. |

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
  Province, China, with one manually verified identity per clip. Released for research use in
  `artifacts/farm_frames_528.zip` (see `DATA_LICENSE.md`). The clips were recorded with the farm owner's permission.

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
* The farm evaluation is a **single-session, within-clip probe**: gallery and query frames of an identity
  come from the same recording, so the contrast measures within-clip discrimination, not cross-session or
  cross-day generalisation.
* No on-device latency, energy or thermal measurements were made.

---

## Citation

See `CITATION.cff`. If you use the farm frames, please also cite the dataset record (DOI assigned on
publication).
