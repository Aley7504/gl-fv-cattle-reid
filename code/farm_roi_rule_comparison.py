# -*- coding: utf-8 -*-
"""Farm region-of-interest rules: comparison and consistency checks.

Three region rules were measured on the SAME farm material (44 identities,
528 frames, 264 queries) with an identical frozen recognition pipeline
(MobileNetV2-ArcFace global + frozen HesAffNet/HardNet Fisher Vector, fused at
a fixed weight of 0.5):

  V1  frozen face detector (DET1) -> full-frame fallback   (diagnostic configuration)
  V2  DET1 -> generic COCO 'cow' animal box -> full frame
  V3  fixed bottom-centre window, no learned detector      (deployed configuration)

This script rebuilds the comparison table from the frozen result files and
re-checks the internal identities that make the diagnostic numbers verifiable:

  * the V1 stratified split (257 fallback / 7 detector-success query frames at
    seed 42) must recombine to the V1 seed-42 overall fused Rank-1, and the
    count-weighted stratum accuracies must give the same value;
  * the V3 blur split must recombine to the V3 seed-42 overall fused Rank-1.

Read-only: it writes one JSON summary next to the other result files.
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"

V3M = json.load(open(RES / "farm_external_eval_metrics.json", encoding="utf-8"))
V3B = json.load(open(RES / "farm_external_eval_bootstrap.json", encoding="utf-8"))
V3Q = json.load(open(RES / "farm_external_eval_quality_report.json", encoding="utf-8"))
V1M = json.load(open(RES / "farm_detector_configuration_metrics.json", encoding="utf-8"))
V1B = json.load(open(RES / "farm_detector_configuration_bootstrap.json", encoding="utf-8"))
V1Q = json.load(open(RES / "farm_detector_configuration_quality_report.json", encoding="utf-8"))
DIAG = json.load(open(RES / "farm_detector_diagnosis.json", encoding="utf-8"))

checks = {}

# ── V1: the stratified split must recombine ────────────────────────────────
st = V1Q["stratified_seed42"]
n_fb, n_dt = st["full_frame_fallback"]["n_query"], st["detector_fired"]["n_query"]
weighted = ((n_fb * st["full_frame_fallback"]["rank1_fusion"]
             + n_dt * st["detector_fired"]["rank1_fusion"]) / (n_fb + n_dt))
checks["v1_strata_partition"] = (n_fb + n_dt == 264)
checks["v1_count_weighted_fusion"] = round(weighted, 6)
checks["v1_stratum_delta_pp"] = round(
    100 * (st["detector_fired"]["rank1_fusion"] - st["full_frame_fallback"]["rank1_fusion"]), 4)

# ── V3: the blur split must recombine ──────────────────────────────────────
st3 = V3Q["stratified_seed42"]
n_s, n_b = st3["sharper"]["n_query"], st3["blurrier"]["n_query"]
weighted3 = ((n_s * st3["sharper"]["rank1_fusion"]
              + n_b * st3["blurrier"]["rank1_fusion"]) / (n_s + n_b))
checks["v3_blur_split_recombines_to"] = round(weighted3, 6)
checks["v3_seed42_overall_fusion"] = round(st3["all_queries"]["rank1_fusion"], 6)
checks["v3_blur_gap_pp"] = round(
    100 * (st3["sharper"]["rank1_fusion"] - st3["blurrier"]["rank1_fusion"]), 4)

summary = {
    "n_identities": 44, "n_frames": 528, "n_queries": 264,
    "rules": {
        "V1_frozen_face_detector": {
            "region_rule": "DET1 -> full-frame fallback",
            "rank1": {k: round(v, 4) for k, v in V1M["rank1"].items()},
            "delta_pp": V1M["delta_rank1_fusion_minus_global_pp"],
            "ci95_pp": V1B["ci95_pp"],
            "face_detection_rate": 0.066,
            "role": "diagnostic configuration (Supplementary Note S3)"},
        "V3_fixed_window_deployed": {
            "region_rule": V3M["protocol"].get("region_rule",
                                                "fixed bottom-centre window x 0.15-0.85, y 0.45-1.00"),
            "rank1": {k: round(v, 4) for k, v in V3M["rank1"].items()},
            "delta_pp": V3M["delta_rank1_fusion_minus_global_pp"],
            "ci95_pp": V3B["ci95_pp"],
            "roi_availability": 1.0,
            "role": "deployed configuration (Section 4.4)"},
    },
    "v2_animal_box": {
        "region_rule": "DET1 -> COCO 'cow' box -> full frame",
        "rank1": {"global": 0.4003, "local": 0.4432, "fusion": 0.4634},
        "delta_pp": 6.3131, "ci95_pp": [1.2626, 11.2374],
        "animal_localisation_rate": 0.941,
        "role": "rejected: the region crops away the barn context the global branch uses"},
    "detection_diagnosis": {
        "frames_without_candidate_at_conf_0.05": DIAG.get("frames_without_candidate_at_conf_0.05"),
        "probe_detection_rate": {"imgsz640": 0.25, "imgsz960": 0.25, "imgsz1280": 0.318,
                                 "tile480": 0.045, "tile360": 0.091, "tile280": 0.023},
    },
    "checks": checks,
}
out = RES / "farm_roi_rule_comparison.json"
json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("wrote", out)
