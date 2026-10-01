"""Inference: finger-tapping video -> kinematics -> MDS-UPDRS 3.4 score + PD probability.

    python -m handtap.predict clip.mp4 --json result.json --plot result.png

    from handtap.predict import HandTapPredictor
    res = HandTapPredictor().predict("clip.mp4")      # JSON-serialisable dict

Research prototype, not a medical device: the models were trained on 234 clips from one
hospital (HUBU-FIS) and have not been validated elsewhere.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import MODELS, __version__
from .kinematics import FEATURES, QC, analyse
from .pose import extract_keypoints

CLASSIFIERS = MODELS / "classifier"

# name -> (label, unit, one-line meaning) for display in the app
FEATURE_INFO = {
    "n_taps": ("Taps", "count", "opening-closing cycles detected"),
    "tap_rate_hz": ("Tap rate", "Hz", "taps per second over the tapping period"),
    "iti_mean_s": ("Inter-tap interval", "s", "mean time between successive openings"),
    "iti_median_s": ("Inter-tap interval (median)", "s", ""),
    "iti_cv": ("Rhythm variability", "CV", "coefficient of variation of inter-tap intervals"),
    "iti_slope_pct": ("Slowing", "%/tap", "trend of inter-tap interval (+ = slowing down)"),
    "open_speed_mean": ("Opening speed", "PL/s", "peak opening speed, palm lengths per second"),
    "close_speed_mean": ("Closing speed", "PL/s", "peak closing speed"),
    "speed_mean": ("Mean speed", "PL/s", "mean absolute aperture speed"),
    "speed_cv": ("Speed variability", "CV", ""),
    "speed_slope_pct": ("Speed trend", "%/tap", "- = speed decrement"),
    "speed_decrement": ("Speed decrement", "ratio-1", "last third vs first third of taps"),
    "open_close_ratio": ("Open/close time ratio", "", ""),
    "duty_cycle": ("Opening share of cycle", "", ""),
    "amp_mean": ("Amplitude", "PL", "mean opening amplitude in palm lengths"),
    "amp_median": ("Amplitude (median)", "PL", ""),
    "amp_max": ("Max amplitude", "PL", ""),
    "amp_cv": ("Amplitude variability", "CV", ""),
    "amp_slope_pct": ("Amplitude trend", "%/tap", "- = amplitude decrement (sequence effect)"),
    "amp_decrement": ("Amplitude decrement", "ratio-1", "last third vs first third of taps"),
    "amp_first5": ("Amplitude, first 5 taps", "PL", ""),
    "amp_last5_ratio": ("Last-5 / first-5 amplitude", "", ""),
    "peak_aperture_mean": ("Peak aperture", "PL", ""),
    "min_aperture_mean": ("Closure gap", "PL", "aperture left at closure (incomplete taps)"),
    "angle_amp_mean": ("Angular amplitude", "deg", "thumb-wrist-index angle excursion"),
    "n_hesitations": ("Hesitations", "count", "intervals > 2x the median interval"),
    "hesitation_rate": ("Hesitation rate", "per interval", ""),
    "n_halts": ("Halts", "count", ">= 0.5 s without aperture movement mid-sequence"),
    "halt_time_s": ("Halt time", "s", ""),
    "iti_max_over_median": ("Longest pause", "x median", ""),
    "dom_freq_hz": ("Dominant frequency", "Hz", "Welch spectrum peak of aperture"),
    "spectral_peak_ratio": ("Rhythmicity", "", "power share near the dominant frequency"),
    "spectral_entropy": ("Spectral entropy", "", "higher = less rhythmic"),
    "sparc": ("Smoothness (SPARC)", "", "more negative = less smooth"),
    "vel_peaks_per_tap": ("Velocity peaks per tap", "", "2 = one smooth opening + closing"),
    "wrist_rms_pl": ("Wrist motion", "PL", "RMS wrist displacement"),
    "coverage": ("Hand tracked", "fraction of frames", ""),
    "active_duration_s": ("Tapping duration", "s", ""),
    "amp_mean_mm": ("Amplitude (metric)", "mm", "MediaPipe world-landmark estimate"),
}


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


class HandTapPredictor:
    def __init__(self, classifier_dir=CLASSIFIERS, threads=None, max_side=None, det_every=1,
                 parallel=False, qc=True):
        """max_side / det_every / parallel: speed settings passed to extract_keypoints
        (defaults = the training extraction; see scripts/check_fast_mode.py).
        qc: also run MediaPipe as a cross-check of the tracking (agreement r, mm amplitude).
        It roughly doubles the processing time and does not change the predictions when the
        classifiers use RTMPose features only."""
        self.speed = dict(max_side=max_side, det_every=det_every, parallel=parallel)
        self.bundles = {}
        for task in ("updrs", "pd"):
            f = Path(classifier_dir) / f"handtap_{task}.joblib"
            if f.exists():
                self.bundles[task] = joblib.load(f)
        if not self.bundles:
            raise FileNotFoundError(f"no classifier bundles in {classifier_dir}; run scripts/fit_final.py")
        fs = {b["featureset"] for b in self.bundles.values()}
        # backends the classifiers need; MediaPipe is added on top when qc is on
        self.required = set().union(*({"mediapipe", "rtmpose"} if s == "kin_fusion"
                                      else {s.replace("kin_", "")} for s in fs))
        self.qc = qc
        self.backends = self._backends(qc)
        self.threads = threads

    def _backends(self, qc):
        return sorted(self.required | ({"mediapipe"} if qc else set()))

    def features_for(self, featureset, per_backend):
        if featureset == "kin_fusion":
            a, b = per_backend["mediapipe"], per_backend["rtmpose"]
            return {k: np.nanmean([a[k], b[k]]) if np.isfinite([a[k], b[k]]).any() else np.nan
                    for k in FEATURES}
        return per_backend[featureset.replace("kin_", "")]

    def predict(self, video_path, max_seconds=None, progress=None, keep_signals=True, qc=None,
                return_details=False):
        """Analyse one video. With return_details=True also returns the raw keypoints,
        signals, taps and per-backend features (for the overlay renderer)."""
        t0 = time.time()
        backends = self._backends(self.qc if qc is None else qc)
        raw = extract_keypoints(video_path, backends=backends, max_seconds=max_seconds,
                                threads=self.threads, progress=progress, **self.speed)
        per, sigs, taps = {}, {}, {}
        for b in backends:
            world = raw[b].get("world") if b == "mediapipe" else None
            sigs[b], taps[b], per[b] = analyse(raw[b]["kpts"], raw[b]["valid"], raw["t"], world)
        # the backend the UPDRS model reads drives the displayed metrics, taps and signals
        fs0 = self.bundles.get("updrs", next(iter(self.bundles.values())))["featureset"]
        primary = fs0.replace("kin_", "") if fs0 != "kin_fusion" else "mediapipe"

        out = dict(version=__version__, video=Path(video_path).name,
                   duration_s=_clean(raw["t"][-1] if len(raw["t"]) else 0.0),
                   fps=_clean(raw["fps"]), frames=int(len(raw["t"])),
                   backends=backends, prediction={}, quality={}, kinematics={})

        for task, bun in self.bundles.items():
            f = self.features_for(bun["featureset"], per)
            x = np.array([[f[k] for k in bun["features"]]], float)
            proba = bun["pipeline"].predict_proba(x)[0]
            classes = [int(c) for c in bun["pipeline"].classes_]
            if task == "updrs":
                out["prediction"]["updrs"] = dict(
                    score=int(bun["pipeline"].predict(x)[0]),
                    expected_score=_clean(float(np.dot(proba, classes))),
                    probabilities={str(c): _clean(p) for c, p in zip(classes, proba)},
                    model=bun["model"], featureset=bun["featureset"],
                    cv_metrics=bun.get("cv_metrics", {}))
            else:
                p = float(proba[classes.index(1)])
                out["prediction"]["pd"] = dict(
                    probability=_clean(p), label="Parkinson's disease" if p >= 0.5 else "Control",
                    model=bun["model"], featureset=bun["featureset"],
                    cv_metrics=bun.get("cv_metrics", {}))

        # kinematics shown to the user = exactly what the UPDRS model saw
        shown = self.features_for(fs0, per)
        for k in FEATURES + QC:
            v = shown[k] if k in FEATURES else per[primary].get(k)
            if k == "amp_mean_mm" and "mediapipe" in per:
                v = per["mediapipe"]["amp_mean_mm"]
            lab, unit, desc = FEATURE_INFO.get(k, (k, "", ""))
            out["kinematics"][k] = dict(value=_clean(v), label=lab, unit=unit, description=desc)
        out["kinematics_by_backend"] = {b: {k: _clean(v) for k, v in per[b].items()} for b in per}

        cov = {b: _clean(float(raw[b]["valid"].mean())) for b in backends}
        warnings = []
        if min(v or 0 for v in cov.values()) < 0.8:
            warnings.append("hand tracked in < 80% of frames - keep the whole hand in view, good light")
        if (per[primary]["n_taps"] or 0) < 8:
            warnings.append("fewer than 8 taps detected - ask for ~10 s of continuous tapping")
        if len(sigs) == 2:
            a, b = (sigs[k]["aperture"] for k in backends)
            ok = np.isfinite(a) & np.isfinite(b)
            r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 30 else np.nan
            out["quality"]["backend_agreement_r"] = _clean(r)
            if not np.isfinite(r) or r < 0.9:
                warnings.append("MediaPipe and RTMPose disagree on the finger aperture - results are less reliable")
        out["quality"].update(coverage=cov, warnings=warnings, ok=not warnings)

        out["taps"] = [{k: _clean(v) for k, v in row.items()} for row in taps[primary].to_dict("records")]
        if keep_signals:
            s = sigs[primary]
            step = max(1, int(round(s["fs"] / 30)))
            out["signals"] = dict(backend=primary, t=[_clean(v) for v in s["t"][::step]],
                                  aperture=[_clean(v) for v in s["aperture"][::step]],
                                  velocity=[_clean(v) for v in s["velocity"][::step]])
            if len(sigs) == 2:
                other = [b for b in backends if b != primary][0]
                out["signals"][f"aperture_{other}"] = [_clean(v) for v in sigs[other]["aperture"][::step]]
        out["processing_s"] = round(time.time() - t0, 1)
        out["disclaimer"] = ("Research prototype trained on 234 clips (HUBU-FIS). Not a diagnostic "
                             "device; a clinician must interpret the result.")
        if return_details:
            return out, dict(raw=raw, sigs=sigs, taps=taps, features=per, primary=primary)
        return out


def plot_result(res, path=None):
    """Aperture trace with detected taps + UPDRS probabilities (matplotlib figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["signals"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[4, 1]))
    t = np.array(s["t"], float)
    a1.plot(t, np.array(s["aperture"], float), color="#2a78d6", lw=1.6, label=s["backend"])
    for k, v in s.items():
        if k.startswith("aperture_"):
            a1.plot(t, np.array(v, float), color="#eb6834", lw=1.0, alpha=0.8, label=k[9:])
    if res["taps"]:
        tp = [x["t_peak"] for x in res["taps"]]
        pk = [x["peak_aperture"] for x in res["taps"]]
        a1.plot(tp, pk, "v", color="#0b0b0b", ms=5, label="tap")
    a1.set_xlabel("time (s)")
    a1.set_ylabel("thumb-index aperture (palm lengths)")
    a1.legend(loc="upper right", fontsize=8, frameon=False)
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(color="#e4e3df", lw=0.6)
    if "updrs" in res["prediction"]:
        p = res["prediction"]["updrs"]["probabilities"]
        ks = sorted(p, key=int)
        cols = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
        a2.bar(ks, [p[k] for k in ks], color=cols[:len(ks)], width=0.7)
        a2.set_ylim(0, 1)
        a2.set_xlabel("MDS-UPDRS 3.4")
        a2.set_title(f"predicted {res['prediction']['updrs']['score']}", fontsize=10)
        a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--json")
    ap.add_argument("--plot")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    res = HandTapPredictor(threads=args.threads).predict(args.video)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    if args.plot:
        plot_result(res, args.plot)
    pr = res["prediction"]
    if "updrs" in pr:
        print(f"MDS-UPDRS 3.4 finger tapping: {pr['updrs']['score']} "
              f"(expected {pr['updrs']['expected_score']}, p={pr['updrs']['probabilities']})")
    if "pd" in pr:
        print(f"P(Parkinson's) = {pr['pd']['probability']:.2f} -> {pr['pd']['label']}")
    k = res["kinematics"]
    print(f"taps {k['n_taps']['value']}, rate {k['tap_rate_hz']['value']} Hz, amplitude "
          f"{k['amp_mean']['value']} PL, amp trend {k['amp_slope_pct']['value']} %/tap, "
          f"hesitations {k['n_hesitations']['value']}")
    for w in res["quality"]["warnings"]:
        print("WARNING:", w)
    print(f"({res['processing_s']} s)")


if __name__ == "__main__":
    main()
