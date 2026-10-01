"""Kinematics overlay video: hand skeleton + thumb-index aperture, live aperture trace with
detected taps, and a summary panel. Used by the app and by scripts/render_overlay.py.

    from handtap.overlay import render_overlay
    render_overlay(video, kpts, valid, sig, taps, feat, info, "out.mp4", width=960, stride=2)

The layout is designed at 1280x720 and scaled by width/1280. `stride` renders every n-th
frame (the output plays in real time at fps/stride). Encoding uses imageio-ffmpeg's bundled
ffmpeg (H.264, yuv420p, faststart) so the result plays in any browser.
"""
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from .pose import INDEX_TIP, THUMB_TIP  # noqa: E402
from .video import iter_frames  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
         (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
         (18, 19), (19, 20)]
# reference palette (see scripts/summarize.py): series blue, accent orange, ordinal ramp, inks
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)
ORD = [(134, 182, 239), (57, 135, 229), (28, 92, 171), (13, 54, 107)]
INK, INK2, SURF, GRID = (11, 11, 11), (82, 81, 78), (252, 252, 251), (228, 227, 223)
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"
_FONTS = {}


def _font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(
            str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), max(6, size))
    return _FONTS[key]


def _even(x):
    return int(round(x / 2) * 2)


def _bgr(c):
    return c[::-1]


def _crop_box(k, frame_hw, cw, h):
    """Fixed crop (no jitter) around all keypoints of the clip, cw:h aspect, padded 30 %."""
    pts = k[np.isfinite(k).all(-1)]
    Hf, Wf = frame_hw
    if len(pts) == 0:
        return 0, 0, Wf, Hf
    x0, y0 = np.percentile(pts, 1, 0)
    x1, y1 = np.percentile(pts, 99, 0)
    cx, cy, w, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.3, (y1 - y0) * 1.3
    aspect = cw / h
    w, hh = (w, w / aspect) if w / max(hh, 1e-6) > aspect else (hh * aspect, hh)
    w, hh = min(w, Wf), min(hh, Hf)
    x0 = int(np.clip(cx - w / 2, 0, Wf - w))
    y0 = int(np.clip(cy - hh / 2, 0, Hf - hh))
    return x0, y0, max(1, int(w)), max(1, int(hh))


def _trace_image(sig, taps, width, height, k):
    """Static aperture plot + the pixel mapping needed to draw the playhead on top."""
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=np.array(SURF) / 255)
    ax = fig.add_axes([0.09, 0.2, 0.88, 0.68])
    ax.set_facecolor(np.array(SURF) / 255)
    t, a = sig["t"], sig["aperture"]
    ax.plot(t, a, color=np.array(BLUE) / 255, lw=1.6 * k)
    if len(taps):
        ax.plot(taps.t_peak, taps.peak_aperture, "v", color=np.array(INK) / 255, ms=5 * k)
    ax.set_xlim(t[0], t[-1])
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or hi <= lo:
        lo, hi = 0.0, 1.0
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.15 * (hi - lo))
    ax.set_xlabel("time (s)", fontsize=9 * k, color=np.array(INK2) / 255)
    ax.set_ylabel("aperture (palm lengths)", fontsize=9 * k, color=np.array(INK2) / 255)
    ax.set_title("thumb–index aperture · ▼ detected taps", fontsize=10 * k, loc="left",
                 color=np.array(INK) / 255)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(np.array(INK2) / 255)
    ax.tick_params(labelsize=8 * k, colors=np.array(INK2) / 255)
    ax.grid(color=np.array(GRID) / 255, lw=0.6 * k)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    img = cv2.resize(img, (width, height)) if img.shape[:2] != (height, width) else img
    tr = ax.transData
    (px0, _), (px1, _) = tr.transform([(t[0], 0), (t[-1], 0)])
    ylim = ax.get_ylim()
    (_, py0), (_, py1) = tr.transform([(0, ylim[0]), (0, ylim[1])])
    plt.close(fig)
    span = max(t[-1] - t[0], 1e-6)

    def to_px(tt, aa):  # matplotlib's y axis points up, image rows point down
        return (int(px0 + (tt - t[0]) / span * (px1 - px0)),
                int(height - (py0 + (aa - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))))
    return img, to_px, (int(height - py1), int(height - py0))


def _draw_hand(img, p, k):
    p = p.astype(int)
    lw, r0 = max(1, round(3 * k)), max(2, round(4 * k))
    for a, b in EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), (255, 255, 255), lw + 2, cv2.LINE_AA)
        cv2.line(img, tuple(p[a]), tuple(p[b]), _bgr(BLUE), lw, cv2.LINE_AA)
    cv2.line(img, tuple(p[THUMB_TIP]), tuple(p[INDEX_TIP]), (255, 255, 255), lw + 3, cv2.LINE_AA)
    cv2.line(img, tuple(p[THUMB_TIP]), tuple(p[INDEX_TIP]), _bgr(ORANGE), lw + 1, cv2.LINE_AA)
    for j, q in enumerate(p):
        tip = j in (THUMB_TIP, INDEX_TIP)
        r = round(7 * k) if tip else r0
        cv2.circle(img, tuple(q), r + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(q), r, _bgr(ORANGE) if tip else _bgr(BLUE), -1, cv2.LINE_AA)


def _fmt(v, spec, default="–"):
    try:
        return format(float(v), spec) if v is not None and np.isfinite(float(v)) else default
    except (TypeError, ValueError):
        return default


def _panel(draw, x, y, w, k, feat, taps, t_now, info):
    S = lambda v: int(round(v * k))  # noqa: E731
    draw.text((x, y), info.get("title", ""), font=_font(S(22), True), fill=INK)
    draw.text((x, y + S(30)), info.get("subtitle", ""), font=_font(S(14)), fill=INK2)
    y += S(70)
    done = taps[taps.t_end <= t_now] if len(taps) else taps
    draw.text((x, y), f"tap {len(done)} / {len(taps)}", font=_font(S(18), True), fill=INK)
    if len(done):
        last = done.iloc[-1]
        draw.text((x + S(160), y + S(2)), f"last tap: amplitude {last.amplitude:.2f} PL · "
                  f"open {last.open_speed:.1f} / close {last.close_speed:.1f} PL/s",
                  font=_font(S(14)), fill=INK2)
    y += S(36)
    rows = [("Tap rate", f"{_fmt(feat.get('tap_rate_hz'), '.2f')} Hz"),
            ("Amplitude", f"{_fmt(feat.get('amp_mean'), '.2f')} PL (CV {_fmt(feat.get('amp_cv'), '.2f')})"),
            ("Amplitude trend", f"{_fmt(feat.get('amp_slope_pct'), '+.2f')} %/tap"),
            ("Mean speed", f"{_fmt(feat.get('speed_mean'), '.2f')} PL/s (CV {_fmt(feat.get('speed_cv'), '.2f')})"),
            ("Rhythm CV", _fmt(feat.get("iti_cv"), ".2f")),
            ("Hesitations / halts", f"{_fmt(feat.get('n_hesitations'), '.0f')} / {_fmt(feat.get('n_halts'), '.0f')}")]
    for i, (key, val) in enumerate(rows):
        cx, cy = x + (i % 2) * (w // 2), y + (i // 2) * S(26)
        draw.text((cx, cy), key, font=_font(S(13)), fill=INK2)
        draw.text((cx + S(135), cy), val, font=_font(S(13), True), fill=INK)
    y += 3 * S(26) + S(18)
    draw.line((x, y, x + w, y), fill=GRID, width=1)
    y += S(12)
    draw.text((x, y), "MDS-UPDRS 3.4", font=_font(S(15), True), fill=INK)
    draw.text((x + S(150), y + S(1)), info.get("updrs_line", ""), font=_font(S(15)), fill=INK)
    y += S(28)
    probs = info.get("probs")
    if probs:
        bw = (w - S(30)) // 4
        for c, pr in enumerate(probs[:4]):
            bx = x + c * (bw + S(10))
            draw.rectangle((bx, y, bx + bw, y + S(14)), fill=GRID)
            draw.rectangle((bx, y, bx + int(bw * float(pr)), y + S(14)), fill=ORD[c])
            draw.text((bx, y + S(17)), f"{c}: {float(pr):.2f}", font=_font(S(12)), fill=INK2)
    y += S(42)
    draw.text((x, y), info.get("pd_line", ""), font=_font(S(15), True), fill=INK)
    draw.text((x, y + S(24)), info.get("footer", ""), font=_font(S(12)), fill=INK2)


def render_overlay(video_path, kpts, valid, sig, taps, feat, info, out_path, width=1280,
                   stride=1, preset="slow", crf=28, progress=None):
    """Write the overlay MP4 and return its path.

    kpts/valid: (T,21,2)/(T,) keypoints of the displayed backend in full-frame pixels;
    sig/taps/feat: kinematics.analyse() output for them; info: panel text, keys
    title, subtitle, updrs_line, probs (4 floats), pd_line, footer.
    """
    from imageio_ffmpeg import get_ffmpeg_exe
    k = width / 1280
    W, H, CW = _even(width), _even(720 * k), _even(540 * k)
    TH = int(round(330 * k))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_total = len(kpts)
    first = next(iter_frames(video_path))[2]
    x0, y0, cw, ch = _crop_box(kpts, first.shape[:2], CW, H)
    s = CW / cw
    trace, to_px, (ytop, ybot) = _trace_image(sig, taps, W - CW, TH, k)
    fps = float(sig.get("fs", 30.0))
    enc = subprocess.Popen([get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps / stride:.3f}", "-i", "-",
                            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    ap = sig["aperture"]
    t_sig = sig["t"]
    try:
        for i, _, img in iter_frames(video_path):
            if i >= frames_total:
                break
            if i % stride:
                continue
            ti = t_sig[i] if i < len(t_sig) else t_sig[-1]
            crop = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (CW, H), interpolation=cv2.INTER_AREA)
            if valid[i]:
                _draw_hand(crop, (kpts[i] - [x0, y0]) * s, k)
            canvas = np.full((H, W, 3), SURF, np.uint8)
            canvas[:, :CW] = crop[:, :, ::-1]
            panel = trace.copy()
            a_i = ap[i] if i < len(ap) else np.nan
            px, py = to_px(ti, a_i if np.isfinite(a_i) else np.nanmin(ap))
            cv2.line(panel, (px, ytop), (px, ybot), ORANGE, max(1, round(2 * k)), cv2.LINE_AA)
            if np.isfinite(a_i):
                cv2.circle(panel, (px, py), round(6 * k), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(panel, (px, py), round(4 * k), ORANGE, -1, cv2.LINE_AA)
            canvas[:TH, CW:] = panel
            im = Image.fromarray(canvas)
            dr = ImageDraw.Draw(im)
            if np.isfinite(a_i):   # live aperture readout on the video
                S = lambda v: int(round(v * k))  # noqa: E731
                dr.rectangle((S(12), S(12), S(262), S(50)), fill=(255, 255, 255))
                dr.text((S(22), S(19)), f"aperture {a_i:.2f} PL", font=_font(S(20), True), fill=ORANGE)
            _panel(dr, CW + int(28 * k), int(345 * k), W - CW - int(56 * k), k, feat, taps, ti, info)
            enc.stdin.write(np.asarray(im).tobytes())
            if progress and frames_total:
                progress(min(1.0, (i + 1) / frames_total))
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode:
        raise RuntimeError(f"ffmpeg failed with exit code {enc.returncode}")
    return out_path
