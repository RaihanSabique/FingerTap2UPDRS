"""Stage 3: nested cross-validation of every (task, feature set, CV protocol, model).

    python scripts/evaluate.py --list          # print the job table
    python scripts/evaluate.py --job 17        # run one job (Slurm array index)

Tasks
  updrs : 4-class MDS-UPDRS 3.4 finger-tapping score (0-3; no 4s in HUBU-FIS)
  pd    : Parkinson's disease vs control (from the file name: IDxx = PD, CONTROLxx = control)

CV protocols
  subject : 5 x repeated 5-fold StratifiedGroupKFold, grouped by participant so a person's
            left and right hand never sit on both sides of a split. Inner loop: 4-fold
            StratifiedGroupKFold grid search maximising MCC. <- the honest estimate.
  loo     : leave-one-VIDEO-out with inner RepeatedStratifiedKFold(5, 2), exactly the
            UBU-PD-FT-Assessment protocol. The other hand of the same person stays in the
            training set, so this is optimistic; reported only for comparability.

Writes outputs/cv/<task>/<featureset>__<cv>__<model>.csv (one row per test prediction).
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (GridSearchCV, LeaveOneOut, RepeatedStratifiedKFold,
                                     StratifiedGroupKFold)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import ROOT  # noqa: E402
from handtap.kinematics import FEATURES  # noqa: E402
from handtap.models import MODEL_NAMES, SEED, make_model  # noqa: E402

FEAT = ROOT / "outputs/features"
CV_OUT = ROOT / "outputs/cv"
LABELS = ROOT / "data/videos_FIS/fis_diagnostic.csv"

TASKS = ["updrs", "pd"]
FEATURESETS = ["kin_mediapipe", "kin_rtmpose", "kin_vitpose", "kin_fusion",
               "ref_mediapipe", "ref_rtmpose"]
CVS = ["subject", "loo"]
N_REPEATS = 5


def load_labels():
    lab = pd.read_csv(LABELS, dtype={"ID": str}).drop_duplicates().set_index("ID")
    lab["subject"] = lab.index.str.rsplit("_", n=1).str[0]
    lab["hand"] = lab.index.str.rsplit("_", n=1).str[1].map({"DCHA": "right", "IZDA": "left"})
    lab["pd"] = lab.index.str.startswith("ID").astype(int)
    lab["updrs"] = lab["UPDRS"].astype(int)
    return lab


def load_features(fs):
    if fs == "kin_fusion":
        # average the two deployable backends: per-feature consensus of MediaPipe and RTMPose
        a, b = load_features("kin_mediapipe"), load_features("kin_rtmpose")
        return (a + b.loc[a.index, a.columns]) / 2
    df = pd.read_csv(FEAT / f"{fs}.csv", index_col="ID")
    return df[FEATURES] if fs.startswith("kin_") else df


def jobs():
    out = []
    for task, fs, cv, m in itertools.product(TASKS, FEATURESETS, CVS, MODEL_NAMES):
        if m == "ordinal_rf" and task == "pd":
            continue
        out.append((task, fs, cv, m))
    return out


def run(task, fs, cv, model_name, n_jobs):
    lab = load_labels()
    X = load_features(fs)
    ids = X.index.intersection(lab.index)
    X, lab = X.loc[ids].astype(float), lab.loc[ids]
    y, g = lab[task].to_numpy(), lab["subject"].to_numpy()
    Xa = X.to_numpy()
    classes = np.unique(y)
    rows = []
    splits = []
    if cv == "subject":
        for r in range(N_REPEATS):
            sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
            splits += [(r, f, tr, te) for f, (tr, te) in enumerate(sgk.split(Xa, y, g))]
    else:
        splits = [(0, f, tr, te) for f, (tr, te) in enumerate(LeaveOneOut().split(Xa))]
    for r, f, tr, te in splits:
        pipe, grid = make_model(model_name)
        if cv == "subject":
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 100 * r + f)
            gs = GridSearchCV(pipe, grid, scoring="matthews_corrcoef", cv=inner, n_jobs=n_jobs,
                              error_score=0.0)
            gs.fit(Xa[tr], y[tr], groups=g[tr])
        else:
            inner = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
            gs = GridSearchCV(pipe, grid, scoring="matthews_corrcoef", cv=inner, n_jobs=n_jobs,
                              error_score=0.0)
            gs.fit(Xa[tr], y[tr])
        pred = gs.predict(Xa[te])
        proba = gs.predict_proba(Xa[te])
        for i, k in enumerate(te):
            row = dict(ID=ids[k], subject=g[k], repeat=r, fold=f, y_true=int(y[k]),
                       y_pred=int(pred[i]), params=json.dumps(gs.best_params_, default=str))
            for c, cl in enumerate(gs.classes_):
                row[f"p_{cl}"] = float(proba[i, c])
            rows.append(row)
    out = CV_OUT / task
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{fs}__{cv}__{model_name}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--job", type=int, nargs="*")
    ap.add_argument("--cv", default=None, help="only run jobs of this CV protocol")
    ap.add_argument("--n_jobs", type=int, default=8)
    args = ap.parse_args()
    J = jobs()
    if args.list:
        for i, j in enumerate(J):
            print(i, *j)
        sys.exit()
    todo = args.job if args.job else range(len(J))
    for i in todo:
        task, fs, cv, m = J[i]
        if args.cv and cv != args.cv:
            continue
        if (CV_OUT / task / f"{fs}__{cv}__{m}.csv").exists():
            continue
        t0 = time.time()
        run(task, fs, cv, m, args.n_jobs)
        print(f"[{i}] {task} {fs} {cv} {m}: {time.time() - t0:.0f}s", flush=True)
