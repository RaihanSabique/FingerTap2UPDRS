"""Overlay videos of the finger-tapping kinematics from the best backend (RTMPose).

    python scripts/render_overlay.py --ids ID125_DCHA CONTROL01_DCHA
    python scripts/render_overlay.py --per_class 1 --gif     # one clip per clinician score

Each 1280x720 video shows the hand crop with the RTMPose-m hand skeleton and the
thumb-index aperture, the aperture trace with detected taps and a playhead, and the
clip's kinematic summary. The UPDRS score and P(PD) shown are OUT-OF-FOLD predictions from
the participant-grouped CV (repeat 0), i.e. from models that never saw this participant.
Writes docs/media/<ID>_overlay.mp4 (and <ID>_overlay.gif with --gif).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import ROOT  # noqa: E402
from handtap.kinematics import analyse  # noqa: E402
from handtap.pose import INDEX_TIP, THUMB_TIP  # noqa: E402
from handtap.video import iter_frames  # noqa: E402
from evaluate import load_labels  # noqa: E402

BACKEND = "rtmpose"
OUT = ROOT / "docs/media"
W, H, CW = 1280, 720, 540            # canvas, hand-crop width
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
         (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
         (18, 19), (19, 20)]
# reference palette (see summarize.py): series blue, accent orange, ordinal blue ramp, inks
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)
ORD = [(134, 182, 239), (57, 135, 229), (28, 92, 171), (13, 54, 107)]
INK, INK2, SURF, GRID = (11, 11, 11), (82, 81, 78), (252, 252, 251), (228, 227, 223)
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def bgr(c):
    return c[::-1]


def oof_predictions():
    u = pd.read_csv(ROOT / "outputs/cv/updrs/kin_rtmpose__subject__ordinal_rf.csv")
    p = pd.read_csv(ROOT / "outputs/cv/pd/kin_rtmpose__subject__rf.csv")
    u, p = u[u.repeat == 0].set_index("ID"), p[p.repeat == 0].set_index("ID")
    return u, p


def crop_box(k, size):
    """Fixed crop (no jitter) around all keypoints of the clip, 3:4, padded 30 %."""
    pts = k[np.isfinite(k).all(-1)]
    x0, y0 = np.percentile(pts, 1, 0)
    x1, y1 = np.percentile(pts, 99, 0)
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.3, (y1 - y0) * 1.3
    aspect = CW / H
    w, h = (w, w / aspect) if w / h > aspect else (h * aspect, h)
    Hf, Wf = size
    w, h = min(w, Wf), min(h, Hf)
    x0 = int(np.clip(cx - w / 2, 0, Wf - w))
    y0 = int(np.clip(cy - h / 2, 0, Hf - h))
    return x0, y0, int(w), int(h)


def trace_image(sig, taps, width, height):
    """Static aperture plot + the pixel mapping needed to draw the playhead on top."""
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=np.array(SURF) / 255)
    ax = fig.add_axes([0.09, 0.2, 0.88, 0.68])
    ax.set_facecolor(np.array(SURF) / 255)
    t, a = sig["t"], sig["aperture"]
    ax.plot(t, a, color=np.array(BLUE) / 255, lw=1.6)
    if len(taps):
        ax.plot(taps.t_peak, taps.peak_aperture, "v", color=np.array(INK) / 255, ms=5)
    ax.set_xlim(t[0], t[-1])
    lo, hi = np.nanmin(a), np.nanmax(a)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.15 * (hi - lo))
    ax.set_xlabel("time (s)", fontsize=9, color=np.array(INK2) / 255)
    ax.set_ylabel("aperture (palm lengths)", fontsize=9, color=np.array(INK2) / 255)
    ax.set_title("thumb–index aperture · ▼ detected taps", fontsize=10, loc="left",
                 color=np.array(INK) / 255)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(np.array(INK2) / 255)
    ax.tick_params(labelsize=8, colors=np.array(INK2) / 255)
    ax.grid(color=np.array(GRID) / 255, lw=0.6)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    tr = ax.transData
    (px0, _), (px1, _) = tr.transform([(t[0], 0), (t[-1], 0)])
    ylim = ax.get_ylim()
    (_, py0), (_, py1) = tr.transform([(0, ylim[0]), (0, ylim[1])])
    plt.close(fig)
    # matplotlib's y axis points up, image rows point down
    to_px = lambda tt, aa: (int(px0 + (tt - t[0]) / (t[-1] - t[0]) * (px1 - px0)),
                            int(height - (py0 + (aa - ylim[0]) / (ylim[1] - ylim[0]) * (py1 - py0))))
    return img, to_px, (int(height - py1), int(height - py0))


def draw_hand(img, k, valid):
    if not valid:
        return
    p = k.astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), (255, 255, 255), 5, cv2.LINE_AA)
        cv2.line(img, tuple(p[a]), tuple(p[b]), bgr(BLUE), 3, cv2.LINE_AA)
    cv2.line(img, tuple(p[THUMB_TIP]), tuple(p[INDEX_TIP]), (255, 255, 255), 6, cv2.LINE_AA)
    cv2.line(img, tuple(p[THUMB_TIP]), tuple(p[INDEX_TIP]), bgr(ORANGE), 4, cv2.LINE_AA)
    for j, q in enumerate(p):
        r = 7 if j in (THUMB_TIP, INDEX_TIP) else 4
        cv2.circle(img, tuple(q), r + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(q), r, bgr(ORANGE) if j in (THUMB_TIP, INDEX_TIP) else bgr(BLUE), -1, cv2.LINE_AA)


def stats_panel(draw, x, y, w, vid, lab, feat, u, p, taps, t_now):
    hand = "right" if vid.endswith("DCHA") else "left"
    group = "Parkinson's disease" if lab.pd == 1 else "control"
    draw.text((x, y), f"{vid}", font=font(22, True), fill=INK)
    draw.text((x, y + 30), f"{hand} hand · {group} · HUBU-FIS (CC-BY-4.0)", font=font(14), fill=INK2)
    y += 70
    done = taps[taps.t_end <= t_now] if len(taps) else taps
    draw.text((x, y), f"tap {len(done)} / {len(taps)}", font=font(18, True), fill=INK)
    if len(done):
        last = done.iloc[-1]
        draw.text((x + 160, y + 2), f"last tap: amplitude {last.amplitude:.2f} PL · "
                  f"open {last.open_speed:.1f} / close {last.close_speed:.1f} PL/s",
                  font=font(14), fill=INK2)
    y += 36
    rows = [("Tap rate", f"{feat['tap_rate_hz']:.2f} Hz"),
            ("Amplitude", f"{feat['amp_mean']:.2f} PL (CV {feat['amp_cv']:.2f})"),
            ("Amplitude trend", f"{feat['amp_slope_pct']:+.2f} %/tap"),
            ("Mean speed", f"{feat['speed_mean']:.2f} PL/s (CV {feat['speed_cv']:.2f})"),
            ("Rhythm CV", f"{feat['iti_cv']:.2f}"),
            ("Hesitations / halts", f"{int(feat['n_hesitations'])} / {int(feat['n_halts'])}")]
    for i, (k, v) in enumerate(rows):
        cx, cy = x + (i % 2) * (w // 2), y + (i // 2) * 26
        draw.text((cx, cy), k, font=font(13), fill=INK2)
        draw.text((cx + 135, cy), v, font=font(13, True), fill=INK)
    y += 3 * 26 + 18
    draw.line((x, y, x + w, y), fill=GRID, width=1)
    y += 12
    draw.text((x, y), "MDS-UPDRS 3.4", font=font(15, True), fill=INK)
    score = sum(c * float(u.get(f"p_{c}", 0.0)) for c in range(4))
    draw.text((x + 150, y + 1), f"clinician {int(lab.updrs)}  ·  model {int(u.y_pred)} "
              f"(out-of-fold, ordinal score {score:.2f})", font=font(15), fill=INK)
    y += 28
    bw = (w - 30) // 4
    for c in range(4):
        pr = float(u.get(f"p_{c}", 0.0))
        bx = x + c * (bw + 10)
        draw.rectangle((bx, y, bx + bw, y + 14), fill=GRID)
        draw.rectangle((bx, y, bx + int(bw * pr), y + 14), fill=ORD[c])
        draw.text((bx, y + 17), f"{c}: {pr:.2f}", font=font(12), fill=INK2)
    y += 42
    draw.text((x, y), f"P(Parkinson's) = {float(p.p_1):.2f}  (out-of-fold)", font=font(15, True), fill=INK)
    draw.text((x, y + 24), "Bars: share of trees voting each score. Research prototype, not a diagnostic device.", font=font(12), fill=INK2)


def render(vid, lab_row, u, p, gif=False):
    d = np.load(ROOT / f"outputs/pose/{vid}.npz")
    k, valid, t = d[f"{BACKEND}_kpts"], d[f"{BACKEND}_valid"], d["t"]
    sig, taps, feat = analyse(k, valid, t)
    video = next(v for v in (ROOT / "data/videos_FIS/videos").iterdir() if v.stem == vid)
    x0, y0, cw, ch = crop_box(k, tuple(d["size"]))
    s = CW / cw
    trace, to_px, (ytop, ybot) = trace_image(sig, taps, W - CW, 330)
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / f"{vid}_overlay.mp4"
    from imageio_ffmpeg import get_ffmpeg_exe
    fps = float(d["fps"])
    enc = subprocess.Popen([get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps:.3f}", "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "28", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    ap_s = sig["aperture"]
    for i, ti, img in iter_frames(video):
        if i >= len(k):
            break
        crop = img[y0:y0 + ch, x0:x0 + cw]
        crop = cv2.resize(crop, (CW, H), interpolation=cv2.INTER_AREA)
        kk = (k[i] - [x0, y0]) * s
        draw_hand(crop, kk, bool(valid[i]))
        canvas = np.full((H, W, 3), SURF, np.uint8)
        canvas[:, :CW] = crop[:, :, ::-1]
        panel = trace.copy()
        px, py = to_px(ti, ap_s[i] if np.isfinite(ap_s[i]) else np.nanmin(ap_s))
        cv2.line(panel, (px, ytop), (px, ybot), ORANGE, 2, cv2.LINE_AA)
        if np.isfinite(ap_s[i]):
            cv2.circle(panel, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, (px, py), 4, ORANGE, -1, cv2.LINE_AA)
        canvas[:330, CW:] = panel
        im = Image.fromarray(canvas)
        dr = ImageDraw.Draw(im)
        if np.isfinite(ap_s[i]):   # live aperture readout on the video
            dr.rectangle((12, 12, 262, 50), fill=(255, 255, 255))
            dr.text((22, 19), f"aperture {ap_s[i]:.2f} PL", font=font(20, True), fill=ORANGE)
        stats_panel(dr, CW + 28, 345, W - CW - 56, vid, lab_row, feat, u, p, taps, ti)
        enc.stdin.write(np.asarray(im).tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"{mp4} ({mp4.stat().st_size / 1e6:.1f} MB): clinician {int(lab_row.updrs)}, "
          f"OOF {int(u.y_pred)}, P(PD) {float(p.p_1):.2f}, {len(taps)} taps")
    if gif:
        g = OUT / f"{vid}_overlay.gif"
        vf = "fps=10,scale=600:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer"
        subprocess.run([get_ffmpeg_exe(), "-y", "-loglevel", "error", "-t", "6", "-i", str(mp4),
                        "-vf", vf, str(g)], check=True)
        print(f"{g} ({g.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--per_class", type=int, default=0,
                    help="random clips per clinician score with good tracking (MediaPipe-RTMPose r >= 0.97)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    lab = load_labels()
    u, p = oof_predictions()
    ids = list(args.ids or [])
    if args.per_class:
        ag = pd.read_csv(ROOT / "outputs/features/backend_agreement.csv", index_col="ID")
        good = ag.index[ag.r_mediapipe_rtmpose >= 0.97]
        for c in range(4):
            pool = lab.loc[lab.index.intersection(good)]
            pool = pool[pool.updrs == c]
            ids += list(pool.sample(args.per_class, random_state=args.seed + c).index)
    for vid in ids:
        render(vid, lab.loc[vid], u.loc[vid], p.loc[vid], gif=args.gif)
