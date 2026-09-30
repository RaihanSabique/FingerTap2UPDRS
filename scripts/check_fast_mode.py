"""Do the app's speed settings change the kinematic features? (run with 2 CPUs, like a free HF Space)

    python scripts/check_fast_mode.py --n 16

For a stratified sample of videos, re-extracts MediaPipe + RTMPose keypoints under several
speed settings, recomputes the consensus features, and compares them with the training
features (full resolution, detector every frame, serial). Writes
outputs/results/fast_mode_check.csv (per setting: seconds per video, per-feature Spearman r
and median relative difference vs training).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import ROOT  # noqa: E402
from handtap.kinematics import FEATURES, analyse  # noqa: E402
from handtap.pose import extract_keypoints  # noqa: E402
from evaluate import load_features, load_labels  # noqa: E402

SETTINGS = {
    "full_serial": dict(max_side=None, det_every=1, parallel=False),
    "960_det1_par": dict(max_side=960, det_every=1, parallel=True),
    "960_det5_par": dict(max_side=960, det_every=5, parallel=True),
    "720_det5_par": dict(max_side=720, det_every=5, parallel=True),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    lab = load_labels()
    ref = load_features("kin_fusion")
    ids = (lab.groupby("updrs", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), args.n // 4), random_state=0)).index)
    vids = {p.stem: p for p in (ROOT / "data/videos_FIS/videos").iterdir()}
    feats = {k: {} for k in SETTINGS}
    secs = {k: [] for k in SETTINGS}
    for vid in ids:
        for name, kw in SETTINGS.items():
            t0 = time.time()
            r = extract_keypoints(vids[vid], backends=("mediapipe", "rtmpose"), threads=args.threads, **kw)
            per = {b: analyse(r[b]["kpts"], r[b]["valid"], r["t"])[2] for b in ("mediapipe", "rtmpose")}
            secs[name].append(time.time() - t0)
            feats[name][vid] = {k: np.nanmean([per["mediapipe"][k], per["rtmpose"][k]]) for k in FEATURES}
        print(vid, {k: round(v[-1], 1) for k, v in secs.items()}, flush=True)
    rows = []
    for name in SETTINGS:
        F = pd.DataFrame(feats[name]).T.astype(float)
        R = ref.loc[F.index, FEATURES].astype(float)
        row = dict(setting=name, sec_per_video=np.mean(secs[name]))
        for c in FEATURES:
            ok = F[c].notna() & R[c].notna()
            row[f"r_{c}"] = spearmanr(F[c][ok], R[c][ok])[0] if ok.sum() > 3 and R[c][ok].nunique() > 1 else np.nan
            row[f"reldiff_{c}"] = np.median(np.abs(F[c][ok] - R[c][ok]) / (np.abs(R[c][ok]) + 1e-9))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/results/fast_mode_check.csv", index=False)
    rc = [c for c in out.columns if c.startswith("r_")]
    dc = [c for c in out.columns if c.startswith("reldiff_")]
    print(out[["setting", "sec_per_video"]].assign(
        median_r=out[rc].median(1), min_r=out[rc].min(1), median_reldiff=out[dc].median(1)))
    print("worst features per setting:")
    for _, r in out.iterrows():
        print(r.setting, r[rc].astype(float).nsmallest(4).round(3).to_dict())
