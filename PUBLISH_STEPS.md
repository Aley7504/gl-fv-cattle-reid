# Publishing this release — exact steps

Everything is staged and committed locally. Three things must happen outside this machine: creating the
GitHub remote, uploading the data bundle to Zenodo, and back-filling the two DOIs. `gh` is not installed
here and the GitHub connector is disabled, so the pushes below must be run by you (or re-run by the agent
once `gh` is authenticated).

---

## 0. Blocking pre-flight checks

- [ ] **Farm owner consent covers redistribution.** The 44 clips were recorded with permission; confirm that
      permission extends to publishing derived frames. If it only covers internal research use, do **not**
      upload `artifacts/farm_frames_528.zip` publicly — replace it with a contact-based access statement.
- [ ] **Holstein2025 licence** allows redistributing derived weights and features (images are already excluded).
- [ ] **Third-party licence**: `third_party/local_fv/` is vendored from the Norppa reference implementation.
      Confirm its licence permits redistribution and keep `README_PROVENANCE.md` alongside it.

---

## 1. Create the GitHub repository and push

Create an empty repository named `gl-fv-cattle-reid` (no README, no .gitignore), then:

```bash
cd "D:/CowReID_Paper/release/gl-fv-cattle-reid"
git remote add origin https://github.com/<YOUR-USER>/gl-fv-cattle-reid.git
git push -u origin main
git tag -a v1.0.0 -m "Manuscript release"
git push origin v1.0.0
```

If you later install the GitHub CLI and run `gh auth login`, the same can be done non-interactively:

```bash
gh repo create <YOUR-USER>/gl-fv-cattle-reid --public --source=. --remote=origin --push
gh release create v1.0.0 --title "v1.0.0" --notes "Code and frozen artefacts for the manuscript."
```

> Note: 3 checkpoints (29 MB) and the split files are inside the repo, so the clone is ~35 MB.

---

## 2. Upload the data bundle to Zenodo (gets the dataset DOI)

1. Log in to <https://zenodo.org>, **New upload**.
2. Upload `artifacts/farm_frames_528.zip` (110.7 MB).
   SHA-256: `0ca75b361c2719ddb12359d717d7ffe59a41ee9840b3d66bebabb448524f7887`
3. Metadata: title *"CLIP: 528 farm cattle-face frames with verified identity labels (single session)"*;
   authors as in `CITATION.cff`; licence **CC BY-NC 4.0**; keywords *cattle face, re-identification, farm video*.
4. Publish, copy the DOI (`10.5281/zenodo.XXXXXXX`).

Optionally archive the **code** too (Zenodo–GitHub integration), which gives the repository its own DOI.

---

## 3. Back-fill the DOIs in the manuscript

Two statements in the manuscript currently read `TBD`:

- **Data availability** — replace `DOI: TBD` with the Zenodo dataset DOI.
- **Code availability** — replace `https://github.com/<user>/<repo>` and `DOI: TBD` with the real URL and,
  if minted, the archived-code DOI.

Then re-tag the repository (`v1.0.1`) so the released code matches the version cited in the paper.

---

## 4. Verify the published package

```bash
git clone https://github.com/<YOUR-USER>/gl-fv-cattle-reid.git
cd gl-fv-cattle-reid
sha256sum -c checksums.sha256          # all files must report OK
python -c "import json;print(json.load(open('results/absolute_r1_from_frozen.json'))['regimes']['g1']['Global'])"
# expected: {'rank1': 0.3121, ...}
```

The `results/` JSON files are the ground truth for every number in Tables 3-6 and Supplementary
Tables S1-S5; if those hashes verify, the published artefact matches the manuscript.
