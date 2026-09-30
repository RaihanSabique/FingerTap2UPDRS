"""Stage 4: metrics tables, feature statistics and figures from the CV predictions.

    python scripts/summarize.py

Writes outputs/results/{metrics_<task>.csv, feature_stats.csv, *.png, RESULTS.md}.
Subject-CV metrics are computed per repeat on the pooled out-of-fold predictions (as the
reference code pools its LOO predictions) and reported as mean +/- SD over the 5 repeats.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import ROOT  # noqa: E402
from handtap.kinematics import FEATURES  # noqa: E402
from handtap.models import compute_metrics  # noqa: E402
from evaluate import load_features, load_labels  # noqa: E402

CV = ROOT / "outputs/cv"
RES = ROOT / "outputs/results"

# reference palette (dataviz skill): ordinal blue ramp for UPDRS 0-3, categorical slots 1-3
ORD = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 130})

KEY = ["MCC", "F1", "Accuracy", "A_AC", "Precision", "Recall"]
KEY_PD = ["MCC", "F1", "Accuracy", "Precision", "Recall", "Sensitivity", "Specificity", "ROC_AUC"]


def metrics_table(task):
    rows = []
    for f in sorted((CV / task).glob("*.csv")):
        fs, cv, model = f.stem.split("__")
        d = pd.read_csv(f)
        pcols = sorted(c for c in d.columns if c.startswith("p_"))
        per = []
        for _, g in d.groupby("repeat"):
            per.append(compute_metrics(g.y_true, g.y_pred, g[pcols].to_numpy(),
                                       classes=[int(c[2:]) for c in pcols], binary=task == "pd"))
        per = pd.DataFrame(per)
        row = dict(featureset=fs, cv=cv, model=model, n=len(d) // d.repeat.nunique())
        for k in per.columns:
            row[k] = per[k].mean()
            row[k + "_sd"] = per[k].std(ddof=0) if len(per) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def participant_table():
    """PD vs control per participant: mean of both hands' out-of-fold P(PD). Valid only for
    subject CV, where a participant's two videos are always in the same test fold."""
    rows = []
    for f in sorted((CV / "pd").glob("*__subject__*.csv")):
        fs, cv, model = f.stem.split("__")
        d = pd.read_csv(f)
        per = []
        for _, g in d.groupby("repeat"):
            s = g.groupby("subject").agg(y=("y_true", "first"), p=("p_1", "mean"))
            proba = np.c_[1 - s.p, s.p]
            per.append(compute_metrics(s.y, (s.p >= 0.5).astype(int), proba, binary=True))
        per = pd.DataFrame(per)
        row = dict(featureset=fs, cv="subject", model=model, n=len(s))
        for k in per.columns:
            row[k], row[k + "_sd"] = per[k].mean(), per[k].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(m, k):
    v, s = m[k], m.get(k + "_sd", np.nan)
    return f"{v:.3f}" if not np.isfinite(s) else f"{v:.3f} ± {s:.3f}"


def confusion(task, fs, cv, model, ax, title):
    d = pd.read_csv(CV / task / f"{fs}__{cv}__{model}.csv")
    labels = sorted(d.y_true.unique())
    reps = d.repeat.nunique()
    cm = pd.crosstab(d.y_true, d.y_pred).reindex(index=labels, columns=labels, fill_value=0) / reps
    ax.imshow(cm.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f7f9fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]))
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm.iat[i, j]
            ax.text(j, i, f"{v:.1f}" if reps > 1 else f"{int(v)}", ha="center", va="center",
                    color="white" if v > cm.to_numpy().max() * 0.55 else INK, fontsize=10)
    names = ["Control", "PD"] if task == "pd" else [str(x) for x in labels]
    ax.set_xticks(range(len(labels)), names)
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Clinician" if task == "updrs" else "Actual")
    ax.grid(False)
    ax.set_title(title, fontsize=9, color=INK)


def feature_stats(fs="kin_fusion"):
    lab = load_labels()
    X = load_features(fs)
    X, lab = X.loc[X.index.intersection(lab.index)], lab.loc[X.index.intersection(lab.index)]
    rows = []
    for c in FEATURES:
        x = X[c].astype(float)
        ok = x.notna()
        rho, p = spearmanr(x[ok], lab.updrs[ok])
        a, b = x[ok & (lab.pd == 1)], x[ok & (lab.pd == 0)]
        u, pu = mannwhitneyu(a, b)
        rows.append(dict(feature=c, spearman_rho_updrs=rho, p_updrs=p, auc_pd_vs_ctrl=u / (len(a) * len(b)),
                         p_pd=pu, median_ctrl=b.median(), median_pd=a.median(),
                         **{f"median_updrs{k}": x[ok & (lab.updrs == k)].median() for k in range(4)}))
    st = pd.DataFrame(rows)
    for p in ("p_updrs", "p_pd"):   # Benjamini-Hochberg FDR
        r = st[p].rank(method="first")
        q = (st[p] * len(st) / r).sort_values(ascending=False).cummin()
        st["q" + p[1:]] = q.reindex(st.index).clip(upper=1)
    return st.sort_values("p_updrs")


def feature_boxplots(st, fs="kin_fusion", n=8):
    lab = load_labels()
    X = load_features(fs).loc[lab.index]
    top = st.head(n).feature.tolist()
    fig, axs = plt.subplots(2, n // 2, figsize=(3.1 * n // 2, 6))
    for ax, c in zip(axs.ravel(), top):
        data = [X[c][lab.updrs == k].dropna() for k in range(4)]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=2))
        for p, col in zip(bp["boxes"], ORD):
            p.set_facecolor(col)
            p.set_edgecolor(col)
        for k, dd in enumerate(data):
            jitter = np.random.default_rng(k).uniform(-0.15, 0.15, len(dd))
            ax.scatter(np.full(len(dd), k + 1) + jitter, dd, s=8, color=INK2, alpha=0.45, lw=0)
        r = st.set_index("feature").loc[c]
        ax.set_title(f"{c}\nρ = {r.spearman_rho_updrs:+.2f}, q = {r.q_updrs:.1e}", fontsize=9)
        ax.set_xticks([1, 2, 3, 4], ["0", "1", "2", "3"])
        ax.set_xlabel("MDS-UPDRS 3.4")
    fig.suptitle("Kinematic features most associated with the clinician's finger-tapping score "
                 "(MediaPipe + RTMPose consensus)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "feature_boxplots.png")
    plt.close(fig)


def heatmap(tab, task, metric="MCC"):
    t = tab[tab.cv == "subject"].pivot(index="featureset", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(1.1 * t.shape[1] + 2.5, 0.5 * t.shape[0] + 1.5))
    ax.imshow(t.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f7f9fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]), vmin=0, vmax=max(0.01, np.nanmax(t.to_numpy())))
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > np.nanmax(t.to_numpy()) * 0.6 else INK, fontsize=9)
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=30, ha="right")
    ax.set_yticks(range(t.shape[0]), t.index)
    ax.grid(False)
    ax.set_title(f"{'UPDRS 0-3' if task == 'updrs' else 'PD vs control'}: {metric}, subject-grouped nested CV",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / f"heatmap_{metric}_{task}.png")
    plt.close(fig)


def backend_plot():
    ag = pd.read_csv(ROOT / "outputs/features/backend_agreement.csv", index_col="ID")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    pairs = [("mediapipe", "rtmpose"), ("mediapipe", "vitpose"), ("rtmpose", "vitpose")]
    data = [ag[f"r_{a}_{b}"].dropna() for a, b in pairs]
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color="white", lw=2),
                    flierprops=dict(marker="o", ms=3, mfc=INK2, mec="none"))
    for p, col in zip(bp["boxes"], CAT):
        p.set_facecolor(col)
        p.set_edgecolor(col)
    ax.set_yticks([1, 2, 3], [f"{a} vs {b}" for a, b in pairs])
    ax.set_xlabel("Pearson r between aperture signals (per video)")
    ax.set_title("Agreement between hand-pose backends", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "backend_agreement.png")
    plt.close(fig)


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    md = ["# HandTap2PDScore - evaluation results\n",
          "HUBU-FIS finger tapping, 234 videos / 118 participants. Metrics are weighted averages "
          "(as in UBU-PD-FT-Assessment). Subject CV = 5 x 5-fold StratifiedGroupKFold by participant, "
          "mean ± SD over repeats; LOO = leave-one-video-out (reference protocol; optimistic because "
          "the other hand of the same person is in training).\n"]
    best = {}
    for task in ("updrs", "pd"):
        tab = metrics_table(task)
        if tab.empty:
            continue
        tab.to_csv(RES / f"metrics_{task}.csv", index=False)
        heatmap(tab, task)
        keys = KEY if task == "updrs" else KEY_PD
        md.append(f"\n## {'MDS-UPDRS 3.4 score (0-3)' if task == 'updrs' else 'PD vs control'}\n")
        for cv in ("subject", "loo"):
            t = tab[tab.cv == cv]
            if t.empty:
                continue
            # best model per feature set by MCC
            b = t.loc[t.groupby("featureset").MCC.idxmax()].sort_values("MCC", ascending=False)
            md.append(f"\n### {cv} CV - best model per feature set (selected by MCC)\n")
            md.append("| Feature set | Model | " + " | ".join(keys) + " |")
            md.append("|---|---|" + "---|" * len(keys))
            for _, m in b.iterrows():
                md.append(f"| {m.featureset} | {m.model} | " + " | ".join(fmt(m, k) for k in keys) + " |")
            if cv == "subject":
                best[task] = b.iloc[0]
        md.append(f"\n### All models, subject CV (MCC)\n")
        piv = tab[tab.cv == "subject"].pivot(index="featureset", columns="model", values="MCC").round(3)
        md.append(piv.to_markdown())
    pt = participant_table()
    if not pt.empty:
        pt.to_csv(RES / "metrics_pd_participant.csv", index=False)
        b = pt.loc[pt.groupby("featureset").MCC.idxmax()].sort_values("MCC", ascending=False)
        md.append("\n## PD vs control per participant (mean P(PD) of both hands, subject CV)\n")
        md.append("| Feature set | Model | " + " | ".join(KEY_PD) + " |")
        md.append("|---|---|" + "---|" * len(KEY_PD))
        for _, m in b.iterrows():
            md.append(f"| {m.featureset} | {m.model} | " + " | ".join(fmt(m, k) for k in KEY_PD) + " |")
    # confusion matrices of the best subject-CV configuration per task (+ LOO counterpart)
    if best:
        fig, axs = plt.subplots(1, 2 * len(best), figsize=(4.2 * 2 * len(best), 4))
        axs = np.atleast_1d(axs)
        i = 0
        for task, m in best.items():
            for cv in ("subject", "loo"):
                f = CV / task / f"{m.featureset}__{cv}__{m.model}.csv"
                if f.exists():
                    confusion(task, m.featureset, cv, m.model, axs[i],
                              f"{task}: {m.featureset} / {m.model}\n{cv} CV"
                              + (" (mean of 5 repeats)" if cv == "subject" else ""))
                i += 1
        fig.tight_layout()
        fig.savefig(RES / "confusion_matrices.png")
        plt.close(fig)
    st = feature_stats()
    st.to_csv(RES / "feature_stats.csv", index=False)
    feature_boxplots(st)
    backend_plot()
    md.append("\n## Feature associations (MediaPipe + RTMPose consensus features)\n")
    md.append("Spearman ρ with UPDRS, Mann-Whitney AUC PD vs control (AUC > 0.5 = higher in PD), "
              "BH-FDR q-values. Videos, not participants, are the unit (2 hands per person), so "
              "p-values are somewhat anti-conservative.\n")
    md.append(st[["feature", "spearman_rho_updrs", "q_updrs", "auc_pd_vs_ctrl", "q_pd",
                  "median_ctrl", "median_pd"]].round(4).to_markdown(index=False))
    (RES / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:60]))
