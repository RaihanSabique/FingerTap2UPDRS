"""Does the app's inference path reproduce the training pipeline's predictions?

    python scripts/check_inference.py --n 12

For a stratified sample, compares the final models applied to the stored training features
with HandTapPredictor.predict() on the raw video (default settings and the app's fast
settings). The videos are training videos, so this checks consistency, not accuracy.
Writes outputs/results/inference_check.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import ROOT  # noqa: E402
from handtap.predict import HandTapPredictor  # noqa: E402
from evaluate import load_features, load_labels  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    lab = load_labels()
    ids = (lab.groupby("updrs", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), args.n // 4), random_state=1)).index)
    vids = {p.stem: p for p in (ROOT / "data/videos_FIS/videos").iterdir()}
    modes = {"default": HandTapPredictor(threads=args.threads),
             "fast": HandTapPredictor(threads=args.threads, max_side=960, parallel=True)}
    b = modes["default"].bundles
    rows = []
    for vid in ids:
        row = dict(ID=vid, clinician=int(lab.updrs[vid]), pd=int(lab.pd[vid]))
        for task in ("updrs", "pd"):
            X = load_features(b[task]["featureset"]).loc[[vid], b[task]["features"]].astype(float)
            pipe = b[task]["pipeline"]
            row[f"stored_{task}"] = int(pipe.predict(X.to_numpy())[0]) if task == "updrs" else \
                float(pipe.predict_proba(X.to_numpy())[0][list(pipe.classes_).index(1)])
        for name, pr in modes.items():
            r = pr.predict(vids[vid], keep_signals=False)
            row[f"{name}_updrs"] = r["prediction"]["updrs"]["score"]
            row[f"{name}_pd"] = r["prediction"]["pd"]["probability"]
            row[f"{name}_sec"] = r["processing_s"]
        rows.append(row)
        print(row, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "outputs/results/inference_check.csv", index=False)
    for name in modes:
        print(f"{name}: UPDRS identical {np.mean(d[f'{name}_updrs'] == d.stored_updrs):.2f}, "
              f"max |dP(PD)| {np.max(np.abs(d[f'{name}_pd'] - d.stored_pd)):.3f}, "
              f"PD label identical {np.mean((d[f'{name}_pd'] >= .5) == (d.stored_pd >= .5)):.2f}, "
              f"mean {d[f'{name}_sec'].mean():.0f} s/video")
