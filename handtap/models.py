"""Classifier zoo, hyper-parameter grids and evaluation metrics (shared by training and the app)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, matthews_corrcoef, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 12883823  # same seed as the UBU-PD-FT-Assessment config


class OrdinalRF(ClassifierMixin, BaseEstimator):
    """Random-forest regression on the ordinal score, rounded to the nearest class.
    Exploits the order of MDS-UPDRS (0 < 1 < 2 < 3), which pure classifiers ignore."""

    def __init__(self, n_estimators=300, max_depth=None, min_samples_leaf=3,
                 max_features=0.5, random_state=SEED):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_samples_leaf, self.max_features = min_samples_leaf, max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # balance classes the way class_weight='balanced' would
        cnt = {c: np.sum(y == c) for c in self.classes_}
        w = np.array([len(y) / (len(cnt) * cnt[v]) for v in y])
        self.rf_ = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features,
                                         random_state=self.random_state, n_jobs=1)
        self.rf_.fit(X, y, sample_weight=w)
        return self

    def predict_score(self, X):
        return self.rf_.predict(X)

    def predict(self, X):
        s = np.clip(np.rint(self.predict_score(X)), self.classes_.min(), self.classes_.max())
        return self.classes_[np.abs(self.classes_[None, :] - s[:, None]).argmin(1)]

    def predict_proba(self, X):
        # per-tree votes -> class frequencies (a calibrated-enough spread for display/AUC)
        votes = np.stack([np.rint(t.predict(X)) for t in self.rf_.estimators_], 1)
        votes = np.clip(votes, self.classes_.min(), self.classes_.max())
        return np.stack([(votes == c).mean(1) for c in self.classes_], 1)


def _xgb(**kw):
    from xgboost import XGBClassifier
    return XGBClassifier(subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         n_jobs=1, random_state=SEED, verbosity=0, **kw)


def make_model(name):
    """(estimator pipeline, param grid). Median imputation + scaling go inside the
    pipeline so they are fit on training folds only."""
    if name == "logreg":
        clf = LogisticRegression(max_iter=5000, random_state=SEED)
        grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}
    elif name == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=SEED)
        grid = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1],
                "clf__class_weight": [None, "balanced"]}
    elif name == "knn":
        clf = KNeighborsClassifier()
        grid = {"clf__n_neighbors": [3, 5, 7, 9, 11, 15], "clf__weights": ["uniform", "distance"]}
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=SEED, n_jobs=1)
        grid = {"clf__max_depth": [None, 6], "clf__min_samples_leaf": [1, 3, 5],
                "clf__max_features": ["sqrt", 0.3]}
    elif name == "xgb":
        clf = _xgb()
        grid = {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.03, 0.1]}
    elif name == "ordinal_rf":
        clf = OrdinalRF()
        grid = {"clf__max_depth": [None, 6], "clf__min_samples_leaf": [1, 3, 5],
                "clf__max_features": ["sqrt", 0.5]}
    else:
        raise ValueError(name)
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("clf", clf)])
    return pipe, grid


MODEL_NAMES = ["logreg", "svm", "knn", "rf", "xgb", "ordinal_rf"]


# ----------------------------------------------------------------------------- metrics
def acceptable_accuracy(y, p):
    """A-AC: share of predictions within +/-1 of the clinician's score (UBU-PD-FT definition)."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) <= 1))


def compute_metrics(y, p, proba=None, classes=None, binary=False):
    """The requested metric set (MCC, F1, Accuracy, A-AC, Precision, Recall; weighted
    averages like the reference code) plus a few that are more informative on imbalanced
    ordinal data (macro-F1, balanced accuracy, quadratic kappa, MAE, ROC-AUC)."""
    y, p = np.asarray(y), np.asarray(p)
    m = dict(
        MCC=matthews_corrcoef(y, p),
        F1=f1_score(y, p, average="weighted", zero_division=0),
        Accuracy=accuracy_score(y, p),
        Precision=precision_score(y, p, average="weighted", zero_division=0),
        Recall=recall_score(y, p, average="weighted", zero_division=0),
        F1_macro=f1_score(y, p, average="macro", zero_division=0),
        Balanced_Acc=balanced_accuracy_score(y, p),
    )
    if binary:
        tp, tn = np.sum((y == 1) & (p == 1)), np.sum((y == 0) & (p == 0))
        m["Sensitivity"] = tp / max(1, np.sum(y == 1))
        m["Specificity"] = tn / max(1, np.sum(y == 0))
        if proba is not None:
            m["ROC_AUC"] = roc_auc_score(y, proba[:, -1])
    else:
        m["A_AC"] = acceptable_accuracy(y, p)
        m["QWK"] = cohen_kappa_score(y, p, weights="quadratic")
        m["MAE"] = mean_absolute_error(y, p)
        if proba is not None:
            try:
                m["ROC_AUC"] = roc_auc_score(y, proba, multi_class="ovr", average="weighted",
                                             labels=classes)
            except ValueError:
                m["ROC_AUC"] = np.nan
    return {k: float(v) for k, v in m.items()}


__all__ = ["make_model", "MODEL_NAMES", "compute_metrics", "acceptable_accuracy", "OrdinalRF",
           "SEED", "clone"]
