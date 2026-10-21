# Porting the frozen pipeline to your machine

The scripts were frozen against one directory layout. Only three constants need editing; nothing in the
statistics changes.

1. `code/h2025_common.py`
   * `ROOT`      -> where you unpacked this repository (writes go here)
   * `DATA_ROOT` -> the Holstein2025 images, laid out as `<DATA_ROOT>/<identity>/<file>.jpg`
2. `code/stage_h_local_descriptors.py`
   * `WEIGHT_DIR` -> the directory holding `AffNet.pth`, `HardNet++.pth`, `OriNet.pth`
     (public weights from the Norppa reference implementation)
3. `code/farm_detector_diagnosis.py`
   * `DET` -> path to the DET1 YOLOv8n checkpoint used for the farm ROI stage

Feature layout expected by `stage_i_fv_encode.py` and `stage_j_scores.py`:
`features/local/<subset>_descriptors.npz` and `features/fv/<role>_fv_<subset>.npz`.
`stage_h` and `stage_i` create these; run them in that order.
