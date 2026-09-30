"""Stage 5: fit the deployable classifiers on all 234 videos.

    python scripts/fit_final.py                       # pick best by subject-CV MCC
    python scripts/fit_final.py --updrs kin_fusion:ordinal_rf --pd kin_fusion:logreg

Only feature sets the app can compute cheaply on CPU are eligible (MediaPipe, RTMPose,
their consensus). Hyper-parameters are re-tuned on all data with a participant-grouped
grid search. Writes models/classifier/handtap_<task>.joblib with the subject-CV metrics
of the chosen configuration, so the app can show honest performance numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import MODELS, ROOT, __version__  # noqa: E402
from handtap.models import SEED, make_model  # noqa: E402
from evaluate import load_features, load_labels  # noqa: E402

DEPLOYABLE = ["kin_mediapipe", "kin_rtmpose", "kin_fusion"]
RES = ROOT / "outputs/results"


def choose(task):
    tab = pd.read_csv(RES / f"metrics_{task}.csv")
    t = tab[(tab.cv == "subject") & tab.featureset.isin(DEPLOYABLE)]
    return t.sort_values("MCC", ascending=False).iloc[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--updrs", help="featureset:model override")
    ap.add_argument("--pd", help="featureset:model override")
    args = ap.parse_args()
    out = MODELS / "classifier"
    out.mkdir(parents=True, exist_ok=True)
    lab = load_labels()
    for task in ("updrs", "pd"):
        tab = pd.read_csv(RES / f"metrics_{task}.csv")
        override = getattr(args, task)
        if override:
            fs, model = override.split(":")
            row = tab[(tab.cv == "subject") & (tab.featureset == fs) & (tab.model == model)].iloc[0]
        else:
            row = choose(task)
            fs, model = row.featureset, row.model
        X = load_features(fs)
        ids = X.index.intersection(lab.index)
        X, L = X.loc[ids].astype(float), lab.loc[ids]
        pipe, grid = make_model(model)
        gs = GridSearchCV(pipe, grid, scoring="matthews_corrcoef", n_jobs=8,
                          cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED))
        gs.fit(X.to_numpy(), L[task].to_numpy(), groups=L.subject.to_numpy())
        metric_cols = [c for c in row.index if c not in ("featureset", "cv", "model", "n")]
        bundle = dict(
            task=task, featureset=fs, model=model, features=list(X.columns),
            pipeline=gs.best_estimator_, best_params=gs.best_params_,
            classes=[int(c) for c in gs.best_estimator_.classes_],
            cv_protocol="5x5-fold StratifiedGroupKFold by participant, nested grid search (MCC)",
            cv_metrics={k: (None if pd.isna(row[k]) else round(float(row[k]), 4)) for k in metric_cols},
            n_train=len(X), version=__version__,
            training_data="HUBU-FIS finger tapping (Zenodo 17738775), 234 videos, 118 participants",
        )
        joblib.dump(bundle, out / f"handtap_{task}.joblib")
        print(f"{task}: {fs} / {model}  params {gs.best_params_}  "
              f"subject-CV MCC {row.MCC:.3f}  -> {out / f'handtap_{task}.joblib'}")
        # feature importance of the final model (permutation, on training data - descriptive only)
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(gs.best_estimator_, X.to_numpy(), L[task].to_numpy(),
                                    scoring="matthews_corrcoef", n_repeats=20, random_state=SEED, n_jobs=8)
        pd.DataFrame(dict(feature=X.columns, importance=pi.importances_mean, sd=pi.importances_std)) \
            .sort_values("importance", ascending=False) \
            .to_csv(RES / f"final_importance_{task}.csv", index=False)
    (out / "README.json").write_text(json.dumps(dict(version=__version__, note=(
        "sklearn Pipeline bundles; load with joblib; requires handtap.models on the import path")), indent=2))
