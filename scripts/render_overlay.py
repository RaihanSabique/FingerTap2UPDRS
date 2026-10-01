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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handtap import ROOT  # noqa: E402
from handtap.kinematics import analyse  # noqa: E402
from handtap.overlay import render_overlay  # noqa: E402
from evaluate import load_labels  # noqa: E402

BACKEND = "rtmpose"
OUT = ROOT / "docs/media"


def oof_predictions():
    u = pd.read_csv(ROOT / "outputs/cv/updrs/kin_rtmpose__subject__ordinal_rf.csv")
    p = pd.read_csv(ROOT / "outputs/cv/pd/kin_rtmpose__subject__rf.csv")
    u, p = u[u.repeat == 0].set_index("ID"), p[p.repeat == 0].set_index("ID")
    return u, p


def render(vid, lab_row, u, p, gif=False):
    from imageio_ffmpeg import get_ffmpeg_exe
    d = np.load(ROOT / f"outputs/pose/{vid}.npz")
    k, valid, t = d[f"{BACKEND}_kpts"], d[f"{BACKEND}_valid"], d["t"]
    sig, taps, feat = analyse(k, valid, t)
    video = next(v for v in (ROOT / "data/videos_FIS/videos").iterdir() if v.stem == vid)
    probs = [float(u.get(f"p_{c}", 0.0)) for c in range(4)]
    info = dict(
        title=vid,
        subtitle=f"{'right' if vid.endswith('DCHA') else 'left'} hand · "
                 f"{'Parkinson' + chr(39) + 's disease' if lab_row.pd == 1 else 'control'} · HUBU-FIS (CC-BY-4.0)",
        updrs_line=f"clinician {int(lab_row.updrs)}  ·  model {int(u.y_pred)} "
                   f"(out-of-fold, ordinal score {sum(c * q for c, q in enumerate(probs)):.2f})",
        probs=probs,
        pd_line=f"P(Parkinson's) = {float(p.p_1):.2f}  (out-of-fold)",
        footer="Bars: share of trees voting each score. Research prototype, not a diagnostic device.")
    mp4 = render_overlay(video, k, valid, sig, taps, feat, info, OUT / f"{vid}_overlay.mp4",
                         width=1280, stride=1, preset="slow")
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
