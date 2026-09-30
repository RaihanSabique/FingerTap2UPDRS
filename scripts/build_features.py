"""Stage 2: pose npz -> per-video kinematic features, per-tap tables and signals.

    python scripts/build_features.py

Writes
  outputs/features/kin_<backend>.csv   our MDS-UPDRS-motivated kinematic features (+QC cols)
  outputs/features/ref_<backend>.csv   UBU-PD-FT-Assessment 'classical' features (baseline)
  outputs/taps/<backend>/<ID>.csv      one row per tap
  outputs/features/backend_agreement.csv  aperture-signal agreement between backends
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import ROOT  # noqa: E402
from handtap.kinematics import analyse, reference_features  # noqa: E402
from handtap.pose import BACKENDS  # noqa: E402

POSE = ROOT / "outputs/pose"
FEAT = ROOT / "outputs/features"
TAPS = ROOT / "outputs/taps"


def one(path):
    d = np.load(path)
    vid = path.stem
    kin, ref, sigs = {}, {}, {}
    for b in BACKENDS:
        if f"{b}_kpts" not in d:
            continue
        world = d["mediapipe_world"] if b == "mediapipe" else None
        sig, taps, f = analyse(d[f"{b}_kpts"], d[f"{b}_valid"], d["t"], world)
        f["mean_kp_score"] = float(np.nanmean(d[f"{b}_score"][d[f"{b}_valid"]])) if d[f"{b}_valid"].any() else np.nan
        kin[b] = f
        ref[b] = reference_features(d[f"{b}_kpts"], d[f"{b}_valid"], d["t"])
        (TAPS / b).mkdir(parents=True, exist_ok=True)
        taps.to_csv(TAPS / b / f"{vid}.csv", index=False)
        sigs[b] = sig["aperture"]
    agree = {}
    for i, a in enumerate(BACKENDS):
        for b in BACKENDS[i + 1:]:
            if a in sigs and b in sigs:
                x, y = sigs[a], sigs[b]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 30:
                    agree[f"r_{a}_{b}"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
                    agree[f"mad_{a}_{b}"] = float(np.median(np.abs(x[ok] - y[ok])))
    return vid, kin, ref, agree


if __name__ == "__main__":
    FEAT.mkdir(parents=True, exist_ok=True)
    files = sorted(POSE.glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    res = Parallel(n_jobs=16, verbose=0)(delayed(one)(f) for f in files)
    for b in BACKENDS:
        kin = pd.DataFrame({v: k[b] for v, k, _, _ in res if b in k}).T
        ref = pd.DataFrame({v: r[b] for v, _, r, _ in res if b in r}).T
        if len(kin):
            kin.index.name = "ID"
            ref.index.name = "ID"
            kin.to_csv(FEAT / f"kin_{b}.csv")
            ref.to_csv(FEAT / f"ref_{b}.csv")
            print(f"{b}: {len(kin)} videos, median taps {kin.n_taps.median():.0f}, "
                  f"median coverage {kin.coverage.median():.3f}")
    ag = pd.DataFrame({v: a for v, _, _, a in res}).T
    ag.index.name = "ID"
    ag.to_csv(FEAT / "backend_agreement.csv")
    print(ag.describe().T[["mean", "50%", "min"]])
