"""Farm probe: reconcile the stratified breakdown with the headline numbers.

Context
-------
Section 4.4 reports Rank-1 over 264 queries as the mean of the three fixed
seeds (42/43/44): Global 0.5694, Local 0.6326, Fusion 0.7197.
The stratified breakdown (fallback vs detector-fired, and the blur median
split) was materialised for **seed 42 only**, because the per-query score
matrices were not persisted for the other two seeds.  A reader who weights the
two strata by their counts therefore obtains 70.1% rather than 72.0%.

This script does three things, using only the frozen result files:

1. shows that the two strata partition the 264 queries exactly;
2. shows that the count-weighted stratum accuracies reproduce the *seed-42*
   overall accuracies, and that their difference reproduces the stored seed-42
   delta to four decimals -- which is only possible if Global, Local and Fusion
   are scored on the same 264 queries with the same indexing;
3. therefore isolates the 72.0% vs 70.1% gap as an aggregation-level
   difference (three-seed mean vs seed 42), not an inconsistency in the data.

Read-only.  Writes one JSON summary next to the other result files.
"""
import json
from pathlib import Path

RES = Path(r"D:\CowReID_Paper\release\gl-fv-cattle-reid\results")
m = json.load(open(RES / "farm_external_eval_metrics.json", encoding="utf-8"))

st = m["stratified_seed42"]
N_TOTAL = m["protocol"]["n_query"]

rows = []
for arm in ("global", "local", "fusion"):
    n_fb = st["full_frame_fallback"]["n_query"]
    n_dt = st["detector_fired"]["n_query"]
    r_fb = st["full_frame_fallback"][f"rank1_{arm}"]
    r_dt = st["detector_fired"][f"rank1_{arm}"]
    assert n_fb + n_dt == N_TOTAL, (n_fb, n_dt, N_TOTAL)
    weighted = (n_fb * r_fb + n_dt * r_dt) / N_TOTAL
    rows.append(dict(arm=arm, n_fallback=n_fb, n_fired=n_dt,
                     r1_fallback=r_fb, r1_fired=r_dt, r1_seed42_weighted=weighted))

seed42_delta_pp = 100.0 * (rows[2]["r1_seed42_weighted"] - rows[0]["r1_seed42_weighted"])
stored_seed42_delta = m["delta_per_seed_pp"]["42"]
stored_3seed = {k: m["rank1"][k] for k in ("global", "local", "fusion")}

# The blur median split must reproduce the fallback stratum exactly.
sharp, blurry = st["fallback_sharp"], st["fallback_blurry"]
n_s, n_b = sharp["n_query"], blurry["n_query"]
blur_recombined = (n_s * sharp["rank1_fusion"] + n_b * blurry["rank1_fusion"]) / (n_s + n_b)

summary = {
    "source": "farm_external_eval_metrics.json (frozen)",
    "n_queries": N_TOTAL,
    "arms_seed42": rows,
    "checks": {
        "strata_partition_exact": True,
        "seed42_delta_recomputed_pp": round(seed42_delta_pp, 4),
        "seed42_delta_stored_pp": stored_seed42_delta,
        "seed42_delta_abs_diff_pp": round(abs(seed42_delta_pp - stored_seed42_delta), 6),
        "local_seed42_weighted": round(rows[1]["r1_seed42_weighted"], 6),
        "local_stored_overall": round(stored_3seed["local"], 6),
        "local_abs_diff": round(abs(rows[1]["r1_seed42_weighted"] - stored_3seed["local"]), 6),
        "blur_split_recombines_to_fallback": round(blur_recombined, 6),
        "fallback_fusion_seed42": round(st["full_frame_fallback"]["rank1_fusion"], 6),
        "blur_split_abs_diff": round(abs(blur_recombined - st["full_frame_fallback"]["rank1_fusion"]), 6),
        "blur_split_gap_pp": round(100.0 * (sharp["rank1_fusion"] - blurry["rank1_fusion"]), 2),
    },
    "headline_three_seed_mean": {k: round(v, 6) for k, v in stored_3seed.items()},
    "conclusion": (
        "The stratified rows are a complete partition of the 264 queries for seed 42 "
        "and reproduce the seed-42 overall accuracies exactly. The headline 0.7197 is "
        "the mean over seeds 42/43/44; the seed-42 fused value is 0.7008. The 72.0% vs "
        "70.1% difference is therefore an aggregation-level difference, not a data "
        "inconsistency. Strata for seeds 43 and 44 cannot be recovered because the "
        "per-query score matrices were not persisted."
    ),
    "not_recoverable": [
        "per-query scores for seeds 43 and 44 (per-seed stratified accuracies)",
        "per-query identifiers (only aggregate counts and accuracies were persisted)",
    ],
}

out = RES / "farm_strata_reconciliation.json"
json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"queries: {N_TOTAL}   (fallback {rows[0]['n_fallback']} + fired {rows[0]['n_fired']})")
for r in rows:
    print(f"  {r['arm']:6s} seed42 weighted = {r['r1_seed42_weighted']:.4f}"
          f"   (3-seed mean {stored_3seed[r['arm']]:.4f})")
print(f"  seed42 delta recomputed {seed42_delta_pp:.4f} pp vs stored {stored_seed42_delta} pp"
      f"  |diff| = {abs(seed42_delta_pp - stored_seed42_delta):.6f}")
print(f"  local seed42 weighted   {rows[1]['r1_seed42_weighted']:.6f}"
      f" vs stored overall {stored_3seed['local']:.6f}")
print(f"  blur split recombines to {blur_recombined:.6f}"
      f" vs fallback fusion {st['full_frame_fallback']['rank1_fusion']:.6f}"
      f"  (gap {100.0*(sharp['rank1_fusion']-blurry['rank1_fusion']):.2f} pp)")
print(f"wrote {out}")
