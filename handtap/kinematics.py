"""Hand keypoints -> finger-tapping kinematics.

Signal: thumb-tip / index-tip aperture divided by palm length (wrist -> middle MCP, 1 s
rolling median), so it is independent of camera distance. Units are "palm lengths" (PL);
the MediaPipe backend also yields millimetres from its metric world landmarks.

Taps are opening-closing cycles found by a hysteresis detector (see _cycles). The per-video
features follow the MDS-UPDRS 3.4 rating criteria: speed, amplitude, hesitations/halts
and decrement (sequence effect), plus rhythm, spectral and smoothness measures.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

from .pose import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST

LOWPASS_HZ = 7.0      # tapping fundamental is <= ~5 Hz; keeps the sharp closures
MAX_GAP_S = 0.25      # interpolate keypoint dropouts up to this long
HYST_LOW, HYST_HIGH = 0.30, 0.60  # closed / open thresholds on the locally normalised aperture
MIN_TAP_S = 0.10      # at most 10 taps/s
HESITATION_X = 2.0    # inter-tap interval > 2 x median = hesitation/interruption
HALT_S = 0.5          # >= 0.5 s of near-still aperture inside the sequence = halt


# ----------------------------------------------------------------------------- signal
def _fill_gaps(x, valid, max_gap):
    """Linear interpolation across runs of invalid samples no longer than max_gap."""
    x = x.copy()
    idx = np.arange(len(x))
    if valid.sum() < 2:
        return x, valid
    good = np.flatnonzero(valid)
    filled = np.interp(idx, good, x[good])
    bad = ~valid
    # run lengths of the gaps
    edges = np.diff(np.r_[0, bad.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    ok = valid.copy()
    for s, e in zip(starts, ends):
        if e - s <= max_gap and s > 0 and e < len(x):
            x[s:e] = filled[s:e]
            ok[s:e] = True
    return x, ok


def _lowpass(x, fs, fc=LOWPASS_HZ):
    """Zero-phase Butterworth on each contiguous finite run (runs < 15 samples are left raw)."""
    y = x.copy()
    b, a = butter(4, min(fc / (fs / 2), 0.95), "low")
    fin = np.isfinite(x)
    edges = np.diff(np.r_[0, fin.astype(int), 0])
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s >= 15:
            y[s:e] = filtfilt(b, a, x[s:e], padlen=min(3 * max(len(a), len(b)), e - s - 1))
    return y


def build_signals(kpts, valid, t, world=None):
    """Uniformly resampled, gap-filled, filtered aperture / angle / wrist signals."""
    t = np.asarray(t, float)
    fs = 1.0 / np.median(np.diff(t))
    tu = t[0] + np.arange(len(t)) / fs     # uniform grid; phone clips are ~constant rate
    k = kpts.astype(float).copy()
    k[~valid] = np.nan
    gap = int(round(MAX_GAP_S * fs))
    ok = valid.copy()
    for j in range(k.shape[1]):
        for c in range(2):
            k[:, j, c], ok_j = _fill_gaps(k[:, j, c], valid, gap)
    ok = ok_j
    k[~ok] = np.nan

    palm = np.linalg.norm(k[:, MIDDLE_MCP] - k[:, WRIST], axis=1)
    win = max(3, int(round(fs)) | 1)
    palm_s = pd.Series(palm).rolling(win, center=True, min_periods=1).median().to_numpy()
    ap_raw = np.linalg.norm(k[:, THUMB_TIP] - k[:, INDEX_TIP], axis=1) / palm_s
    ap = _lowpass(ap_raw, fs)

    v1 = k[:, INDEX_TIP] - k[:, WRIST]
    v2 = k[:, THUMB_TIP] - k[:, WRIST]
    cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    ang = _lowpass(np.degrees(np.arccos(np.clip(cosang, -1, 1))), fs)

    vel = np.gradient(ap, 1 / fs)
    wrist = _lowpass(k[:, WRIST, 0], fs), _lowpass(k[:, WRIST, 1], fs)
    wrist = np.stack(wrist, 1) / np.nanmedian(palm_s)

    sig = dict(t=tu, fs=fs, aperture=ap, aperture_raw=ap_raw, velocity=vel, angle=ang,
               wrist=wrist, palm_px=palm_s, valid=ok)
    if world is not None:
        w = world.astype(float).copy()
        w[~valid] = np.nan
        mm = np.linalg.norm(w[:, THUMB_TIP] - w[:, INDEX_TIP], axis=1) * 1000
        mm, _ = _fill_gaps(mm, np.isfinite(mm), gap)
        sig["aperture_mm"] = _lowpass(mm, fs)
    return sig


# ----------------------------------------------------------------------------- taps
def _cycles(x, fs):
    """Hysteresis tap detector -> list of (closure_idx, peak_idx, next_closure_idx).

    The aperture is normalised by its local 3 s range (5th-95th percentile, floored at 30 %
    of the global range so halts don't blow up noise). A tap needs the normalised aperture
    to fall below LOW, rise above HIGH and fall below LOW again, so jitter on the open
    plateau or a shallow dip mid-opening never splits a tap, while decrementing
    amplitudes are still followed.
    """
    win = max(3, int(3 * fs))
    s = pd.Series(x)
    lo = s.rolling(win, center=True, min_periods=1).quantile(0.05).to_numpy()
    hi = s.rolling(win, center=True, min_periods=1).quantile(0.95).to_numpy()
    g_lo, g_hi = np.nanpercentile(x, [5, 95])
    rng = np.maximum(hi - lo, max(0.3 * (g_hi - g_lo), 1e-6))
    z = (x - lo) / rng
    # open episodes: from the first sample above HIGH until z drops below LOW again
    episodes, state, start = [], "closed", None
    seen_closed = False
    for i, v in enumerate(z):
        if state == "closed" and v > HYST_HIGH:
            state, start = "open", i
        elif state == "open" and v < HYST_LOW:
            if seen_closed:
                episodes.append((start, i))
            state = "closed"
        if v < HYST_LOW:
            seen_closed = True
    min_len = max(1, int(MIN_TAP_S * fs / 2))
    episodes = [e for e in episodes if e[1] - e[0] >= min_len]
    if not episodes:
        return []
    # closure between consecutive open episodes = argmin of the aperture there
    closures = []
    for k, (a, b) in enumerate(episodes):
        prev_end = episodes[k - 1][1] if k else 0
        closures.append(prev_end + int(np.argmin(x[prev_end:a + 1])))
    out = []
    for k, (a, b) in enumerate(episodes):
        c0 = closures[k]
        if k + 1 < len(episodes):
            c1 = closures[k + 1]
        else:
            # last tap: its closure is the minimum after the episode, if the signal closes
            seg = x[b:min(len(x), b + int(1.5 * fs))]
            if len(seg) == 0:
                continue
            c1 = b + int(np.argmin(seg))
        p = a + int(np.argmax(x[a:b + 1]))
        out.append((c0, p, c1))
    return out


def segment_taps(sig):
    """Find opening-closing cycles. Returns a per-tap DataFrame (may be empty)."""
    ap, t, fs, vel = sig["aperture"], sig["t"], sig["fs"], sig["velocity"]
    if np.isfinite(ap).sum() < 2 * fs:
        return pd.DataFrame()
    # detection only: bridge long tracking gaps; cycles touching a gap are rejected below
    x = pd.Series(ap).interpolate(limit_direction="both").to_numpy()
    rows = []
    for a, p, b in _cycles(x, fs):
        seg = slice(a, b + 1)
        if not np.all(np.isfinite(ap[seg])):
            continue  # a tracking gap inside the cycle: amplitude would be invented
        o, c = slice(a, p + 1), slice(p, b + 1)
        n_vpk = len(find_peaks(np.abs(vel[seg]), prominence=0.1 * np.nanmax(np.abs(vel[seg])))[0])
        rows.append(dict(
            t_start=t[a], t_peak=t[p], t_end=t[b],
            duration=t[b] - t[a], open_time=t[p] - t[a], close_time=t[b] - t[p],
            amplitude=x[p] - x[a], peak_aperture=x[p], min_aperture=x[a],
            open_speed=np.nanmax(vel[o]), close_speed=-np.nanmin(vel[c]),
            mean_speed=np.nanmean(np.abs(vel[seg])),
            vel_peaks=n_vpk,
            amplitude_mm=(sig["aperture_mm"][p] - sig["aperture_mm"][a]) if "aperture_mm" in sig else np.nan,
            angle_amplitude=sig["angle"][p] - sig["angle"][a],
        ))
    taps = pd.DataFrame(rows)
    if len(taps):
        # drop tiny cycles (tremor ripples, tracking jitter) relative to the typical tap
        taps = taps[taps.amplitude >= 0.2 * taps.amplitude.median()].reset_index(drop=True)
        taps.insert(0, "tap", np.arange(1, len(taps) + 1))
    return taps


# ----------------------------------------------------------------------------- features
def _slope_pct(y):
    """Linear trend over the tap sequence as % of the mean per tap (negative = decrement)."""
    y = np.asarray(y, float)
    if len(y) < 3 or not np.isfinite(y).all() or np.mean(y) == 0:
        return np.nan
    return 100 * np.polyfit(np.arange(len(y)), y, 1)[0] / np.mean(y)


def _thirds_ratio(y):
    """mean(last third) / mean(first third) - 1 (negative = decrement)."""
    y = np.asarray(y, float)
    n = len(y) // 3
    if n < 1 or np.mean(y[:n]) == 0:
        return np.nan
    return np.mean(y[-n:]) / np.mean(y[:n]) - 1


def _sparc(speed, fs, fc=10.0, amp_th=0.05, padlevel=4):
    """Spectral arc length (Balasubramanian 2015); more negative = less smooth."""
    speed = speed[np.isfinite(speed)]
    if len(speed) < 16:
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    Mf = Mf / Mf.max()
    sel = f <= fc
    f, Mf = f[sel], Mf[sel]
    inx = np.flatnonzero(Mf >= amp_th)
    if len(inx) == 0:
        return np.nan
    f, Mf = f[inx[0]:inx[-1] + 1], Mf[inx[0]:inx[-1] + 1]
    df, dM = np.diff(f) / (f[-1] - f[0] if f[-1] > f[0] else 1), np.diff(Mf)
    return -np.sum(np.sqrt(df ** 2 + dM ** 2))


def _count_halts(sig, t0, t1, rng):
    ap, t, fs = sig["aperture"], sig["t"], sig["fs"]
    sel = (t >= t0) & (t <= t1) & np.isfinite(ap)
    if sel.sum() < fs:
        return 0, 0.0
    still = np.abs(sig["velocity"]) < 0.5 * rng   # < half a range per second
    still &= sel
    edges = np.diff(np.r_[0, still.astype(int), 0])
    runs = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) / fs
    long = runs[runs >= HALT_S]
    return len(long), float(long.sum())


FEATURES = [
    # speed / rhythm
    "n_taps", "tap_rate_hz", "iti_mean_s", "iti_median_s", "iti_cv", "iti_slope_pct",
    "open_speed_mean", "close_speed_mean", "speed_mean", "speed_cv", "speed_slope_pct",
    "speed_decrement", "open_close_ratio", "duty_cycle",
    # amplitude
    "amp_mean", "amp_median", "amp_max", "amp_cv", "amp_slope_pct", "amp_decrement",
    "amp_first5", "amp_last5_ratio", "peak_aperture_mean", "min_aperture_mean",
    "angle_amp_mean",
    # hesitations / halts
    "n_hesitations", "hesitation_rate", "n_halts", "halt_time_s", "iti_max_over_median",
    # spectral / smoothness
    "dom_freq_hz", "spectral_peak_ratio", "spectral_entropy", "sparc", "vel_peaks_per_tap",
    # proximal / whole-hand motion
    "wrist_rms_pl",
]
QC = ["coverage", "active_duration_s", "amp_mean_mm"]


def video_features(sig, taps):
    """One row of features (dict) from the signals and per-tap table."""
    f = {k: np.nan for k in FEATURES + QC}
    f["coverage"] = float(np.mean(sig["valid"]))
    f["n_taps"] = len(taps)
    if len(taps) < 3:
        f["n_hesitations"], f["n_halts"] = 0, 0
        return f
    t0, t1 = taps.t_start.iloc[0], taps.t_end.iloc[-1]
    dur = t1 - t0
    f["active_duration_s"] = dur
    iti = np.diff(taps.t_peak.to_numpy())
    amp, spd = taps.amplitude.to_numpy(), taps.open_speed.to_numpy() + taps.close_speed.to_numpy()
    f.update(
        tap_rate_hz=len(taps) / dur if dur > 0 else np.nan,
        iti_mean_s=iti.mean(), iti_median_s=np.median(iti), iti_cv=iti.std() / iti.mean(),
        iti_slope_pct=_slope_pct(iti),
        open_speed_mean=taps.open_speed.mean(), close_speed_mean=taps.close_speed.mean(),
        speed_mean=taps.mean_speed.mean(), speed_cv=spd.std() / spd.mean(),
        speed_slope_pct=_slope_pct(spd), speed_decrement=_thirds_ratio(spd),
        open_close_ratio=(taps.open_time / taps.close_time.clip(lower=1e-3)).median(),
        duty_cycle=(taps.open_time / taps.duration).median(),
        amp_mean=amp.mean(), amp_median=np.median(amp), amp_max=amp.max(),
        amp_cv=amp.std() / amp.mean(), amp_slope_pct=_slope_pct(amp),
        amp_decrement=_thirds_ratio(amp), amp_first5=amp[:5].mean(),
        amp_last5_ratio=amp[-5:].mean() / amp[:5].mean() if len(amp) >= 6 else np.nan,
        peak_aperture_mean=taps.peak_aperture.mean(), min_aperture_mean=taps.min_aperture.mean(),
        angle_amp_mean=taps.angle_amplitude.mean(),
        vel_peaks_per_tap=taps.vel_peaks.mean(),
        amp_mean_mm=taps.amplitude_mm.mean(),
    )
    med = np.median(iti)
    f["n_hesitations"] = int(np.sum(iti > HESITATION_X * med))
    f["hesitation_rate"] = f["n_hesitations"] / len(iti)
    f["iti_max_over_median"] = iti.max() / med
    lo, hi = np.nanpercentile(sig["aperture"], [5, 95])
    f["n_halts"], f["halt_time_s"] = _count_halts(sig, t0, t1, hi - lo)

    sel = (sig["t"] >= t0) & (sig["t"] <= t1)
    ap = sig["aperture"][sel]
    ap = ap[np.isfinite(ap)]
    fs = sig["fs"]
    if len(ap) >= 2 * fs:
        fr, P = welch(ap - ap.mean(), fs=fs, nperseg=min(len(ap), int(4 * fs)))
        band = (fr >= 0.5) & (fr <= 8)
        fr, P = fr[band], P[band]
        if P.sum() > 0:
            k = np.argmax(P)
            f["dom_freq_hz"] = fr[k]
            f["spectral_peak_ratio"] = P[np.abs(fr - fr[k]) <= 0.35].sum() / P.sum()
            p = P / P.sum()
            f["spectral_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    f["sparc"] = _sparc(np.abs(sig["velocity"][sel]), fs)
    w = sig["wrist"][sel]
    w = w[np.isfinite(w).all(1)]
    if len(w) > 2:
        f["wrist_rms_pl"] = float(np.sqrt(((w - w.mean(0)) ** 2).sum(1).mean()))
    return f


# ----------------------------------------------------------------------------- reference baseline
def reference_features(kpts, valid, t):
    """Re-implementation of the 'classical' feature set of arelraptor/UBU-PD-FT-Assessment
    (lib/extract_features.py) on 2-D keypoints, used as a like-for-like baseline.
    Their amplitude is thumb-index distance / wrist-index-tip distance with a 3-frame mean."""
    from scipy.stats import kurtosis, skew
    k = kpts[valid].astype(float)
    tt = np.asarray(t)[valid]
    if len(k) < 30:
        return {}
    fps = 1 / np.median(np.diff(tt))
    ang = []
    for fr in k:
        v1, v2 = fr[WRIST] - fr[INDEX_TIP], fr[WRIST] - fr[THUMB_TIP]
        ang.append(np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
    ang = np.array(ang) / 90
    amp = np.linalg.norm(k[:, THUMB_TIP] - k[:, INDEX_TIP], axis=1) / np.linalg.norm(k[:, WRIST] - k[:, INDEX_TIP], axis=1)
    amp = pd.Series(amp).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    amp = amp.rolling(3, center=True).mean().bfill().ffill().to_numpy()
    vel = np.r_[0, np.diff(amp) * fps]
    acc = np.r_[0, np.diff(vel) * fps]
    pk, _ = find_peaks(amp, distance=int(fps // 3))
    per = np.diff(pk / fps)
    freq = np.full(len(amp), np.nan)
    if len(pk) > 1:
        freq[pk[1:]] = 1 / per
    half = len(amp) // 2

    def fft2(x):
        n = len(x)
        F = np.abs(np.fft.fft(x))[: n // 2]
        return np.fft.fftfreq(n)[: n // 2][np.argmax(F)], np.sum(F ** 2)

    fa, pa = fft2(amp)
    fv, pv = fft2(vel)
    return {
        "SMOOTHED_AMPLITUDE_mean": amp.mean(), "SMOOTHED_AMPLITUDE_std": amp.std(ddof=1),
        "SMOOTHED_AMPLITUDE_max": amp.max(), "VELOCITY_mean": vel.mean(), "VELOCITY_max": vel.max(),
        "VELOCITY_std": vel.std(ddof=1), "ACCELERATION_mean": acc.mean(), "ACCELERATION_max": acc.max(),
        "ACCELERATION_std": acc.std(ddof=1), "DISTANCE_ANG_mean": ang.mean(), "DISTANCE_ANG_std": ang.std(ddof=1),
        "FREQUENCY_mean": np.nanmean(freq) if np.isfinite(freq).any() else np.nan,
        "FREQUENCY_std": np.nanstd(freq, ddof=1) if np.isfinite(freq).sum() > 1 else np.nan,
        "FREQUENCY_count": int(np.isfinite(freq).sum()),
        "INTERVAL_STD": per.std() if len(per) >= 2 else np.nan,
        "INTERVAL_CV": per.std() / per.mean() if len(per) >= 2 else np.nan,
        "AMPLITUDE_SLOPE": np.polyfit(np.arange(len(amp)), amp, 1)[0],
        "FREQUENCY_DROP": np.nanmean(freq[:half]) - np.nanmean(freq[-half:]),
        "ACCELERATION_INCREASE": acc[:half].mean() - acc[-half:].mean(),
        "SKEW_AMPLITUDE": skew(amp), "KURTOSIS_AMPLITUDE": kurtosis(amp),
        "SKEW_VELOCITY": skew(vel), "KURTOSIS_VELOCITY": kurtosis(vel),
        "FFT_DOMINANT_FREQ_AMPLITUDE": fa, "FFT_POWER_AMPLITUDE": pa,
        "FFT_DOMINANT_FREQ_VELOCITY": fv, "FFT_POWER_VELOCITY": pv,
    }


def analyse(kpts, valid, t, world=None):
    """Full kinematic analysis of one backend's keypoints -> (signals, taps, features)."""
    sig = build_signals(kpts, valid, t, world)
    taps = segment_taps(sig)
    return sig, taps, video_features(sig, taps)
